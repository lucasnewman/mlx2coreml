import unittest

from mlx2coreml.reporting import init_stage_timings, summarize_stage_timings, timed_stage


class ReportingStageTimingTests(unittest.TestCase):
    def test_timed_stage_records_elapsed_seconds(self) -> None:
        timings = init_stage_timings(["capture", "lower"])
        with timed_stage(timings, "capture"):
            sum(range(1000))

        self.assertIsNotNone(timings["capture"])
        self.assertIsNone(timings["lower"])
        self.assertGreaterEqual(float(timings["capture"]), 0.0)

    def test_timed_stage_records_elapsed_when_exception_is_raised(self) -> None:
        timings = init_stage_timings(["convert"])
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with timed_stage(timings, "convert"):
                raise RuntimeError("boom")

        self.assertIsNotNone(timings["convert"])
        self.assertGreaterEqual(float(timings["convert"]), 0.0)

    def test_summarize_stage_timings_returns_total_and_preserves_missing(self) -> None:
        timings = {"a": 0.1234567, "b": None, "c": 0.4444444}
        summarized, total = summarize_stage_timings(timings, ndigits=4)
        self.assertEqual(summarized["a"], 0.1235)
        self.assertIsNone(summarized["b"])
        self.assertEqual(summarized["c"], 0.4444)
        self.assertEqual(total, 0.5679)


if __name__ == "__main__":
    unittest.main()
