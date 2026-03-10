import unittest

from mlx2coreml.ir import StateSpec


class IrStateSpecTests(unittest.TestCase):
    def test_state_spec_to_dict(self) -> None:
        spec = StateSpec(name="kv_cache", shape=(1, 32, 128, 64), dtype="fp16")
        self.assertEqual(
            spec.to_dict(),
            {"name": "kv_cache", "shape": [1, 32, 128, 64], "dtype": "fp16"},
        )


if __name__ == "__main__":
    unittest.main()
