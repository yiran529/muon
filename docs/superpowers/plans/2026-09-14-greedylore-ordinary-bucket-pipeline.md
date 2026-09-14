# GreedyLore Ordinary Bucket Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline ordinary compressed GreedyLore score preparation and overlap reconstruction of bucket A with communication of bucket B without changing collective order.

**Architecture:** Split the current full-bucket tail into a collective ordering tail and a non-scheduling aggregate completion tail. Prepare ordinary compressed score buffers on a separate CUDA stream at bucket readiness, release the next bucket after factor reduction, and reconstruct on a separate stream before completing the DDP Future.

**Tech Stack:** Python 3.10, PyTorch 2.11 Futures, DDP communication hooks, CUDA streams/events, NCCL/Gloo tests, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-greedylore-ordinary-bucket-pipeline-design.md`

## Global Constraints

- Preserve the exact rank-wide collective sequence `score(A), factor(A), score(B), factor(B)`.
- Do not change GreedyLore gradient, error-feedback, basis, support, or checkpoint tensor semantics.
- Keep warmup and refresh bucket scheduling conservative in this pass.
- Keep prepared buffers alive until the DDP-facing bucket Future completes.
- Do not add a process group or run a formal performance experiment.
- Preserve the user's unrelated `docs/compressed_muon/PAPER_NOTES.md` modification.

---

### Task 1: Split scheduling and completion state

**Files:**
- Modify: `dion/greedy_lore_ddp_hook.py`
- Test: `tests/test_greedy_lore_ddp_state.py`
- Test: `tests/test_greedy_lore_ddp_future.py`

**Interfaces:**
- Produces: `BucketContext.previous_collective_tail`, `BucketContext.collective_completion_future`, `GreedyLoreDDPState.collective_tail`, and an aggregate `GreedyLoreDDPState.tail_future` that remains the lifecycle/checkpoint boundary.

- [x] Write a failing test with two bucket contexts whose second DDP completion resolves before the first; assert `finish_step()` still reports an in-flight tail until both complete.
- [x] Run the focused test and confirm it fails because the current state tracks only the latest completion.
- [x] Implement a non-scheduling aggregate completion Future while installing the collective placeholder before callbacks can run inline.
- [x] Run the state and Future tests and confirm they pass.

### Task 2: Prepare ordinary compressed buckets before the collective tail

**Files:**
- Modify: `dion/greedy_lore_ddp_hook.py`
- Test: `tests/test_greedy_lore_ddp_future.py`
- Test: `tests/test_greedy_lore_ddp_hook.py`

**Interfaces:**
- Produces: `PreparedCompressedBucket`, `GreedyLoreDDPState.preparation_stream(device)`, `_prepare_compressed_bucket(state, context)`, and a launch path consuming the retained prepared object.

- [x] Write a failing test that holds bucket A's completion pending, invokes bucket B's ordinary compressed hook, and observes B's prepared score before A completes while observing no B collective launch.
- [x] Run the focused test and confirm it fails because preparation currently occurs inside the serialized launch callback.
- [x] Add the prepared state object, preparation stream/event handoff, and split preparation from collective launch.
- [x] Run the focused Future and hook tests and confirm they pass.

### Task 3: Release the next collective before reconstruction completion

**Files:**
- Modify: `dion/greedy_lore_ddp_hook.py`
- Test: `tests/test_greedy_lore_ddp_future.py`
- Test: `tests/test_greedy_lore_ddp_hook_nccl.py`

**Interfaces:**
- Produces: `GreedyLoreDDPState.reconstruction_stream(device)` and ordinary compressed factor completion that resolves `collective_completion_future` separately from `completion_future`.

- [x] Write a failing controlled-Future test where A's factor reduction completes but A reconstruction remains pending; assert B's score launch occurs and both DDP Futures remain pending as appropriate.
- [x] Run it and confirm it fails because B currently waits for A reconstruction.
- [x] Queue reconstruction independently, resolve the collective placeholder after factor reduction, and preserve final CUDA visibility before resolving the DDP Future.
- [x] Extend failure tests so collective failure poisons both paths while reconstruction failure does not reorder or suppress later collectives.
- [x] Run all GreedyLore hook, state, Future, distributed Gloo, checkpoint, layout, and training-entry tests.

### Task 4: Record the implementation attempt

**Files:**
- Modify: `docs/worklog/M002-greedy-lore-muon.md`

**Interfaces:**
- Produces: a dated record of code changes, verification commands, limitations, and the deferred formal experiment.

- [x] Append the implementation summary and exact verification evidence in Chinese.
- [x] Inspect `git diff --check` and `git diff --stat`.
- [x] Re-run the final targeted test suite and report any skipped multi-GPU verification explicitly.
