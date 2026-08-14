import pytest

from gr00t.configs.data.data_config import ModalityConfig
from gr00t.configs.smpl_root_mode import (
    action_mode_from_processor,
    assert_processor_kwargs_match_setup,
    infer_root_process_mode_from_processor,
    patch_modality_configs_for_root_mode,
    resolve_smpl_root_training,
)

UNITREE_G1_SMPL = "unitree_g1_smpl"


def _minimal_modality_configs():
    return {
        UNITREE_G1_SMPL: {
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["left_hand", "right_hand", "robot_root", "robot_qpos"],
            ),
            "action": ModalityConfig(
                delta_indices=list(range(50)),
                modality_keys=["frame", "left_hand", "right_hand"],
            ),
        }
    }


def _setup(mode, action_mode=None, **legacy):
    return resolve_smpl_root_training(
        root_process_mode=mode,
        action_mode=action_mode,
        legacy_use_relative_euler=legacy.get("legacy_use_relative_euler", False),
        legacy_use_state_euler=legacy.get("legacy_use_state_euler", False),
    )


def _saved_processor(*, use_relative_euler: bool, use_state_euler: bool, state_keys: list[str]):
    return {
        "use_relative_euler": use_relative_euler,
        "use_state_euler": use_state_euler,
        "modality_configs": {
            UNITREE_G1_SMPL: {"state": {"modality_keys": state_keys}},
        },
    }


def test_original_mode_flags_and_no_state_root():
    setup = _setup("original")
    assert setup.use_relative_euler is False
    assert setup.use_state_euler is False
    assert setup.include_state_robot_root is False
    patched = patch_modality_configs_for_root_mode(
        _minimal_modality_configs(),
        UNITREE_G1_SMPL,
        setup,
    )
    assert "robot_root" not in patched[UNITREE_G1_SMPL]["state"].modality_keys


def test_rot6d_mode():
    setup = _setup("rot6d", action_mode="relative")
    assert setup.use_relative_euler is False
    assert setup.use_state_euler is False
    assert setup.include_state_robot_root is True
    assert setup.action_mode == "relative"


def test_rot6d_rejects_absolute():
    with pytest.raises(ValueError, match="invalid for root-process-mode=rot6d"):
        _setup("rot6d", action_mode="absolute")


def test_delta_euler_mode():
    setup = _setup("delta_euler", action_mode="relative")
    assert setup.use_relative_euler is True
    assert setup.use_state_euler is True
    assert setup.action_mode == "relative"


def test_delta_euler_rejects_absolute():
    with pytest.raises(ValueError, match="invalid for root-process-mode=delta_euler"):
        _setup("delta_euler", action_mode="absolute")


def test_euler_relative_and_absolute():
    rel = _setup("euler", action_mode="relative")
    assert rel.use_relative_euler is True and rel.use_state_euler is True
    abs_ = _setup("euler", action_mode="absolute")
    assert abs_.use_relative_euler is True and abs_.use_state_euler is False


def test_legacy_env_maps_to_delta_euler():
    setup = _setup(
        "original",
        legacy_use_relative_euler=True,
        legacy_use_state_euler=True,
    )
    assert setup.root_process_mode == "delta_euler"


def test_legacy_env_maps_to_euler_absolute():
    setup = _setup(
        "original",
        legacy_use_relative_euler=True,
        legacy_use_state_euler=False,
    )
    assert setup.root_process_mode == "euler"
    assert setup.action_mode == "absolute"


def test_infer_mode_from_processor():
    rot6d_proc = _saved_processor(
        use_relative_euler=False,
        use_state_euler=False,
        state_keys=["robot_root", "robot_qpos"],
    )
    assert infer_root_process_mode_from_processor(rot6d_proc) == "rot6d"
    assert action_mode_from_processor(rot6d_proc) == "relative"

    delta_proc = _saved_processor(
        use_relative_euler=True,
        use_state_euler=True,
        state_keys=["robot_root", "robot_qpos"],
    )
    assert infer_root_process_mode_from_processor(delta_proc) == "delta_euler"


@pytest.mark.parametrize(
    ("saved", "setup"),
    [
        (
            _saved_processor(
                use_relative_euler=False,
                use_state_euler=False,
                state_keys=["robot_root", "robot_qpos"],
            ),
            _setup("rot6d", action_mode="relative"),
        ),
        (
            _saved_processor(
                use_relative_euler=True,
                use_state_euler=True,
                state_keys=["robot_root", "robot_qpos"],
            ),
            _setup("delta_euler"),
        ),
    ],
)
def test_resume_accepts_matching_setup(saved, setup):
    assert_processor_kwargs_match_setup(saved, setup)


def test_resume_rejects_wrong_mode():
    saved = _saved_processor(
        use_relative_euler=True,
        use_state_euler=True,
        state_keys=["robot_root", "robot_qpos"],
    )
    setup = _setup("euler", action_mode="absolute")
    with pytest.raises(ValueError, match="Resume root-process-mode mismatch"):
        assert_processor_kwargs_match_setup(saved, setup)
