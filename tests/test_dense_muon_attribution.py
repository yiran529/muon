import json

import pytest

from benchmark.compressed_muon.dense_muon_attribution import (
    MODE_SPECS,
    DiagnosticConfig,
    diagnostic_benchmark_config,
    mode_spec,
    summarize_artifacts,
)


def test_attribution_exposes_five_isolation_modes_in_single_variable_order():
    assert tuple(MODE_SPECS) == (
        "rank_local_custom_hook",
        "process_group_custom_hook",
        "rank_local_default_reducer",
        "rank_local_no_triton",
        "upstream_ddp",
    )
    assert mode_spec("rank_local_custom_hook") == {
        "distributed_mesh": None,
        "register_custom_ddp_hook": True,
        "use_triton": None,
    }
    assert mode_spec("process_group_custom_hook") == {
        "distributed_mesh": "ddp_process_group",
        "register_custom_ddp_hook": True,
        "use_triton": None,
    }
    assert mode_spec("rank_local_default_reducer") == {
        "distributed_mesh": None,
        "register_custom_ddp_hook": False,
        "use_triton": None,
    }
    assert mode_spec("rank_local_no_triton") == {
        "distributed_mesh": None,
        "register_custom_ddp_hook": True,
        "use_triton": False,
    }
    assert mode_spec("upstream_ddp") == {
        "distributed_mesh": "ddp_process_group",
        "register_custom_ddp_hook": False,
        "use_triton": None,
    }


def test_diagnostic_config_maps_to_existing_benchmark_without_changing_workload():
    config = DiagnosticConfig(mode="rank_local_no_triton")
    benchmark_config = diagnostic_benchmark_config(config, output="cell/result.json")
    assert benchmark_config.optimizer == "muon"
    assert benchmark_config.sync == "dense"
    assert benchmark_config.model == "gpt130m"
    assert benchmark_config.warmup_steps == 2
    assert benchmark_config.measure_steps == 12
    assert benchmark_config.world_size == 4
    assert benchmark_config.sequence_length == 256
    assert benchmark_config.muon_distributed_mesh == "none"
    assert benchmark_config.register_custom_ddp_hook is True
    assert benchmark_config.muon_use_triton is False
    assert benchmark_config.output == "cell/result.json"


@pytest.mark.parametrize(
    ("mode", "mesh", "hook"),
    [
        ("process_group_custom_hook", "ddp_process_group", True),
        ("upstream_ddp", "ddp_process_group", False),
    ],
)
def test_mesh_and_hook_axes_are_independently_selectable(mode, mesh, hook):
    config = diagnostic_benchmark_config(DiagnosticConfig(mode=mode))
    assert config.muon_distributed_mesh == mesh
    assert config.register_custom_ddp_hook is hook


def test_diagnostic_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown attribution mode"):
        DiagnosticConfig(mode="not-a-mode")


def test_summary_keeps_failed_cells_and_reads_rank_checksum_pairs(tmp_path):
    root = tmp_path / "dense_muon_attribution"
    good = root / "rank_local_custom_hook"
    bad = root / "rank_local_default_reducer"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "status.json").write_text(json.dumps({"status": "completed", "exit_code": 0}))
    (good / "result.json").write_text(json.dumps({
        "mode": "rank_local_custom_hook",
        "correctness": {
            "parameter_checksum_agreement": True,
            "rank_checksum_pairs": [[1.0, 2.0], [1.0, 2.0]],
        },
    }))
    (bad / "status.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
    summary = summarize_artifacts(root)
    assert summary["status_counts"] == {"completed": 1, "failed": 1}
    assert summary["cells"][0]["mode"] == "rank_local_custom_hook"
    assert summary["cells"][0]["rank_checksum_pairs"] == [[1.0, 2.0], [1.0, 2.0]]
    assert summary["cells"][1]["result"] is None
