from pathlib import Path

import pytest

from quant_for_fun.examples.run_finrl_2026 import PRESETS
from quant_for_fun.examples.run_finrl_2026 import parse_agent_list
from quant_for_fun.examples.run_finrl_2026 import resolve_path


def test_parse_agent_list_normalizes_supported_agents() -> None:
    assert parse_agent_list("PPO, sac,td3") == ["ppo", "sac", "td3"]


def test_parse_agent_list_rejects_unknown_agents() -> None:
    with pytest.raises(ValueError, match="Unsupported agents"):
        parse_agent_list("ppo,not_real")


def test_quick_preset_is_small_enough_for_first_run() -> None:
    assert PRESETS["quick"].timesteps < PRESETS["release"].timesteps
    assert PRESETS["quick"].universe == "quick"


def test_resolve_path_keeps_external_projects_isolated() -> None:
    project_root = Path("/tmp/quant_for_fun")
    assert resolve_path("external_projects/FinRL", project_root) == (
        project_root / "external_projects/FinRL"
    )
