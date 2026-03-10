import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx2coreml

from mlx2coreml.compilation import compile_mlmodel


class CompilationApiTests(unittest.TestCase):
    def test_compile_mlmodel_copies_compiled_directory_to_requested_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "model.mlpackage"
            model_path.mkdir()

            compiled_temp = temp_path / "compiled-temp"
            compiled_temp.mkdir()
            (compiled_temp / "weights.bin").write_text("compiled", encoding="utf-8")

            compiled_out = temp_path / "artifacts" / "model.mlmodelc"
            compiled_out.parent.mkdir(parents=True, exist_ok=True)
            compiled_out.mkdir()
            (compiled_out / "stale.txt").write_text("stale", encoding="utf-8")

            with patch("mlx2coreml.compilation.ct.models.utils.compile_model") as compile_mock:
                compile_mock.return_value = str(compiled_temp)
                result = compile_mlmodel(model_path, compiled_out)

            self.assertEqual(result, compiled_out.resolve())
            self.assertFalse((compiled_out / "stale.txt").exists())
            self.assertEqual((compiled_out / "weights.bin").read_text(encoding="utf-8"), "compiled")

    def test_compile_mlmodel_is_exported_from_top_level_module(self) -> None:
        self.assertIs(mlx2coreml.compile_mlmodel, compile_mlmodel)


if __name__ == "__main__":
    unittest.main()
