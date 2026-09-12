"""Training-entry contracts for Rand-K and Top-K Muon."""

import argparse
import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    pytest.importorskip("train", reason="requires the dion[train] extra")
    return importlib.import_module("train_sparsek")


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(8, 8, bias=False)
        self.transformer.wte = torch.nn.Embedding(16, 8)
        self.transformer.wte.scale = torch.nn.Parameter(torch.ones(8))
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


class DDPStub:
    process_group = None
    find_unused_parameters = False

    def __init__(self):
        self.registrations = []

    def register_comm_hook(self, state, hook):
        self.registrations.append((state, hook))


def _cli(**overrides):
    values = dict(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
        _explicit_replicate_mesh_grad_sync=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("method", ["randk", "topk"])
def test_entry_builds_ordinary_muon_and_registers_sparse_k_hook(method):
    module = _module()
    import train
    from dion.muon import Muon
    from dion.sparse_k_ddp_hook import SparseKDDPState

    model = StubModel()
    ddp = DDPStub()
    optimizer, runtime = module.init_sparse_k_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.SparseKHyperparameters(sparse_k_method=method),
        cli_args=_cli(),
    )

    assert type(optimizer) is Muon
    assert runtime.optimizer_owns_gradient_sync is False
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert isinstance(state, SparseKDDPState)
    assert state.config.method == method
    assert hook is module.sparse_k_ddp_hook
    assert runtime.begin_step == state.begin_step
    assert runtime.finish_step == state.finish_step
    assert runtime.commit_step == state.commit_step
    assert runtime.checkpoint_state is state
    assert runtime.checkpoint_state_name == "sparse_k_compressor"
    assert train.extra_stateful_from_gradient_sync_runtime(runtime) == {
        "sparse_k_compressor": state
    }


def test_entry_compresses_all_2d_parameters_and_keeps_vectors_dense():
    module = _module()
    model = StubModel()
    ddp = DDPStub()

    module.init_sparse_k_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.SparseKHyperparameters(),
        cli_args=_cli(),
    )

    state, _hook = ddp.registrations[0]
    roles = {spec.stable_name: spec.role for spec in state.parameter_specs}
    assert roles == {
        "transformer.h.weight": "sparse_matrix",
        "transformer.wte.weight": "sparse_matrix",
        "transformer.wte.scale": "dense_aux",
        "lm_head.weight": "sparse_matrix",
    }


def test_entry_rejects_non_ddp_and_legacy_sync_ownership():
    module = _module()
    hp = module.SparseKHyperparameters()

    with pytest.raises(ValueError, match="DDP"):
        module.init_sparse_k_optimizer(None, object(), None, hp, _cli())
    with pytest.raises(ValueError, match="replicate_mesh_grad_sync"):
        module.init_sparse_k_optimizer(
            StubModel(),
            None,
            DDPStub(),
            hp,
            _cli(_explicit_replicate_mesh_grad_sync=True),
        )


def test_parser_exposes_sparse_k_options_and_validation_rejects_wrong_optimizer():
    module = _module()
    parser = argparse.ArgumentParser()
    module.configure_sparse_k_parser(parser)
    args = parser.parse_args(
        [
            "--sparse_k_method",
            "randk",
            "--sparse_k_ratio",
            "0.125",
            "--sparse_k_error_feedback",
            "noef",
            "--sparse_k_seed",
            "9",
            "--sparse_k_start_compress_step",
            "12",
        ]
    )
    assert vars(args) == {
        "sparse_k_method": "randk",
        "sparse_k_ratio": 0.125,
        "sparse_k_error_feedback": "noef",
        "sparse_k_seed": 9,
        "sparse_k_start_compress_step": 12,
    }
    with pytest.raises(ValueError, match="optimizer"):
        module.validate_sparse_k_hyperparameters(
            module.SparseKHyperparameters(optimizer="muon")
        )
