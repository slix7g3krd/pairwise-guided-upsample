import pytest
import imaging.guided_upsample as mod

CASES = [
    ("joint_bilateral_guide", (1,)),
    ("radius_in_output_scale", (1,)),
    ("wait_result_per_image", (1,)),
    ("cache_per_image", (1,)),
    ("mirror_pad", (1,)),
    ("gradient_halo_correct", (1,)),
    ("load", (1,)),
    ("upsample", (1,)),
    ("refine", (1,)),
    ("export", (1,)),
    ("validate_upsample", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
