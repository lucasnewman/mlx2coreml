import unittest
from collections import Counter

from mlx2coreml.compute_plan import _device_label, _sorted_counter_items


class ComputePlanHelperTests(unittest.TestCase):
    def test_device_label_maps_known_device_type_names(self) -> None:
        cpu = type("MLCPUComputeDevice", (), {})()
        gpu = type("MLGPUComputeDevice", (), {})()
        ane = type("MLNeuralEngineComputeDevice", (), {})()
        other = type("SomethingElse", (), {})()

        self.assertEqual(_device_label(cpu), "CPU")
        self.assertEqual(_device_label(gpu), "GPU")
        self.assertEqual(_device_label(ane), "ANE")
        self.assertEqual(_device_label(other), "SomethingElse")
        self.assertEqual(_device_label(None), "UNKNOWN")

    def test_sorted_counter_items_orders_by_count_then_name(self) -> None:
        counts = Counter({"zeta": 1, "alpha": 3, "beta": 3, "gamma": 2})
        ranked = _sorted_counter_items(counts, top_k=3)
        self.assertEqual(ranked, [["alpha", 3], ["beta", 3], ["gamma", 2]])


if __name__ == "__main__":
    unittest.main()
