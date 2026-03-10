import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx2coreml.lower_to_mil import MultifunctionBinding, save_multifunction_package


class MultifunctionPackagingHelperTests(unittest.TestCase):
    @patch("mlx2coreml.lower_to_mil.ct.models.MLModel")
    @patch("mlx2coreml.lower_to_mil.ct.utils.save_multifunction")
    @patch("mlx2coreml.lower_to_mil.ct.utils.MultiFunctionDescriptor")
    def test_save_multifunction_package_adds_bindings_and_metadata(
        self,
        mock_descriptor_ctor,
        mock_save_multifunction,
        mock_mlmodel_ctor,
    ) -> None:
        descriptor = MagicMock()
        mock_descriptor_ctor.return_value = descriptor
        mlmodel = MagicMock()
        mlmodel.user_defined_metadata = {}
        mock_mlmodel_ctor.return_value = mlmodel

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "source.mlpackage"
            destination = Path(temp_dir) / "dest.mlpackage"
            save_multifunction_package(
                bindings=[
                    MultifunctionBinding(
                        model_path=model_path,
                        source_function="main",
                        target_function="infer",
                    )
                ],
                destination_path=destination,
                default_function_name="infer",
                metadata={"mx2mil.test": {"ok": True}},
            )

            descriptor.add_function.assert_called_once()
            args, _ = descriptor.add_function.call_args
            self.assertEqual(args[1], "main")
            self.assertEqual(args[2], "infer")
            self.assertEqual(descriptor.default_function_name, "infer")
            mock_save_multifunction.assert_called_once()
            mock_mlmodel_ctor.assert_called_once_with(str(destination.resolve()), skip_model_load=True)
            self.assertIn("mx2mil.test", mlmodel.user_defined_metadata)
            mlmodel.save.assert_called_once()
            save_args, _ = mlmodel.save.call_args
            self.assertIn(".tmp_meta", save_args[0])

    def test_save_multifunction_package_rejects_empty_bindings(self) -> None:
        with self.assertRaises(ValueError):
            save_multifunction_package(
                bindings=[],
                destination_path=Path("dummy.mlpackage"),
                default_function_name="main",
            )


if __name__ == "__main__":
    unittest.main()
