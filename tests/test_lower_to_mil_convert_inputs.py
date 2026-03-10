import unittest
from unittest.mock import patch

import coremltools as ct

from mlx2coreml.ir import StateSpec
from mlx2coreml.lower_to_mil import convert_program_to_model


class LowerToMilConvertInputsTests(unittest.TestCase):
    @patch("mlx2coreml.lower_to_mil.ct.convert")
    def test_convert_program_to_model_passes_optional_inputs(self, mock_convert) -> None:
        sentinel_program = object()
        sentinel_inputs = ["input_spec_0"]
        convert_program_to_model(
            sentinel_program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_precision="fp16",
            compute_units="cpu_only",
            inputs=sentinel_inputs,
        )

        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertEqual(args[0], sentinel_program)
        self.assertEqual(kwargs["inputs"], sentinel_inputs)
        self.assertEqual(kwargs["compute_units"], ct.ComputeUnit.CPU_ONLY)
        self.assertEqual(kwargs["compute_precision"], ct.precision.FLOAT16)

    @patch("mlx2coreml.lower_to_mil.ct.convert")
    def test_convert_program_to_model_builds_states_from_state_specs(self, mock_convert) -> None:
        sentinel_program = object()
        convert_program_to_model(
            sentinel_program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            state_specs=[StateSpec(name="kv_cache", shape=(1, 2, 3), dtype="fp16")],
        )

        mock_convert.assert_called_once()
        _, kwargs = mock_convert.call_args
        self.assertIn("states", kwargs)
        states = kwargs["states"]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].name, "kv_cache")
        self.assertEqual(tuple(states[0].wrapped_type.shape.shape), (1, 2, 3))

    @patch("mlx2coreml.lower_to_mil.ct.convert")
    def test_convert_program_to_model_retries_without_states_for_mil_program(self, mock_convert) -> None:
        sentinel_program = object()
        sentinel_model = object()
        mock_convert.side_effect = [
            ValueError("'states' can only be passed with pytorch source model."),
            sentinel_model,
        ]

        converted = convert_program_to_model(
            sentinel_program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            state_specs=[StateSpec(name="kv_cache", shape=(1, 2, 3), dtype="fp16")],
        )

        self.assertIs(converted, sentinel_model)
        self.assertEqual(mock_convert.call_count, 2)
        first_kwargs = mock_convert.call_args_list[0].kwargs
        second_kwargs = mock_convert.call_args_list[1].kwargs
        self.assertIn("states", first_kwargs)
        self.assertNotIn("states", second_kwargs)

    def test_convert_program_to_model_rejects_both_states_and_state_specs(self) -> None:
        with self.assertRaises(ValueError):
            convert_program_to_model(
                object(),
                states=[object()],
                state_specs=[StateSpec(name="kv_cache", shape=(1, 2), dtype="fp16")],
            )


if __name__ == "__main__":
    unittest.main()
