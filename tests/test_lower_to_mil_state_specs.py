import unittest

from mlx2coreml.ir import StateSpec
from mlx2coreml.lower_to_mil import build_coreml_state_types


class LowerToMilStateSpecTests(unittest.TestCase):
    def test_build_coreml_state_types_creates_named_state_types(self) -> None:
        states = build_coreml_state_types(
            [StateSpec(name="kv_cache", shape=(1, 2, 3), dtype="fp16")]
        )
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertEqual(state.name, "kv_cache")
        self.assertEqual(tuple(state.wrapped_type.shape.shape), (1, 2, 3))

    def test_build_coreml_state_types_rejects_unsupported_dtype(self) -> None:
        with self.assertRaises(ValueError):
            build_coreml_state_types(
                [StateSpec(name="bad_state", shape=(1, 2), dtype="fp64")]
            )


if __name__ == "__main__":
    unittest.main()
