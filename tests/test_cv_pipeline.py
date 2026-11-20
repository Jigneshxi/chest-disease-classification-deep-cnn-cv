"""
Comprehensive Automated Acceptance & Regression Test Suite for M2_DLCV
Verifies dataset integrity, PyTorch CNN architecture, forward pass, inference engine,
benchmark evaluation metrics, and notebook execution status.
"""

import os
import sys
import json
import unittest
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import ChestXRayCNN, CVInferenceEngine

class TestComputerVisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = BASE_DIR
        cls.data_dir = os.path.join(BASE_DIR, "data")
        cls.models_dir = os.path.join(BASE_DIR, "models")
        cls.model_weights = os.path.join(cls.models_dir, "best_cv_model.pt")
        cls.class_names_path = os.path.join(cls.models_dir, "class_names.json")
        cls.metrics_path = os.path.join(cls.models_dir, "evaluation_metrics.json")
        cls.sample_dir = os.path.join(cls.data_dir, "sample_images")
        cls.notebook_path = os.path.join(BASE_DIR, "Sample CV Submission Template.ipynb")
        cls.engine = CVInferenceEngine(model_dir=cls.models_dir)

    def test_01_dataset_directory_structure_and_counts(self):
        """Verify train, val, and test splits contain 4 class subdirectories and 1,200 total images."""
        expected_splits = {"train": 840, "val": 180, "test": 180}
        total_found = 0
        for split, expected_count in expected_splits.items():
            split_dir = os.path.join(self.data_dir, split)
            self.assertTrue(os.path.exists(split_dir), f"Directory {split} missing.")
            classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            self.assertEqual(len(classes), 4, f"Expected 4 classes in {split}, found {len(classes)}")
            split_files = [
                os.path.join(split_dir, c, f) 
                for c in classes 
                for f in os.listdir(os.path.join(split_dir, c)) 
                if f.endswith('.png')
            ]
            self.assertEqual(len(split_files), expected_count, 
                             f"Split {split} count mismatch: expected {expected_count}, got {len(split_files)}")
            total_found += len(split_files)
        self.assertEqual(total_found, 1200, f"Expected 1,200 total images, found {total_found}")

    def test_02_class_names_registry(self):
        """Verify class names JSON exists and contains the 4 target pathology classes."""
        self.assertTrue(os.path.exists(self.class_names_path), "class_names.json missing.")
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        expected = ["Bacterial_Pneumonia", "COVID19", "Normal", "Viral_Pneumonia"]
        self.assertEqual(sorted(classes), sorted(expected))

    def test_03_sample_radiology_images_exist(self):
        """Verify presence of all 4 representative sample images for testing."""
        for cname in ["Normal", "COVID19", "Bacterial_Pneumonia", "Viral_Pneumonia"]:
            spath = os.path.join(self.sample_dir, f"sample_{cname}.png")
            self.assertTrue(os.path.exists(spath), f"Sample image missing: {spath}")
            with Image.open(spath) as img:
                self.assertEqual(img.size, (224, 224), f"Sample image size {img.size} is not 224x224")

    def test_04_cnn_architecture_and_parameters(self):
        """Validate Deep CNN layer design and parameter volume."""
        model = ChestXRayCNN(num_classes=4)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 1000000, "Model has insufficient parameter capacity.")
        layers = list(model.features.children())
        self.assertIsInstance(layers[0], torch.nn.Conv2d)
        self.assertEqual(layers[0].in_channels, 3)
        self.assertEqual(layers[0].out_channels, 32)

    def test_05_forward_pass_tensor_dimensions(self):
        """Test forward pass with batch tensor and verify probability axioms."""
        batch_size = 4
        dummy_tensor = torch.randn(batch_size, 3, 224, 224)
        self.engine.model.eval()
        with torch.no_grad():
            logits = self.engine.model(dummy_tensor)
            probs = torch.softmax(logits, dim=1)
        self.assertEqual(logits.shape, (batch_size, 4))
        self.assertEqual(probs.shape, (batch_size, 4))
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)

    def test_06_model_checkpoint_weights_load(self):
        """Verify checkpoint file exists and loads weights without exception."""
        self.assertTrue(os.path.exists(self.model_weights), "best_cv_model.pt missing.")
        state = torch.load(self.model_weights, map_location='cpu')
        self.assertIsInstance(state, dict)
        self.assertIn("features.0.weight", state)
        self.assertIn("classifier.4.weight", state)

    def test_07_inference_engine_single_image_prediction(self):
        """Verify single image prediction, confidence bounds, and clinical recommendation."""
        sample_path = os.path.join(self.sample_dir, "sample_Normal.png")
        result = self.engine.predict(sample_path)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["predicted_class"], "Normal")
        self.assertGreaterEqual(result["confidence_score"], 0.90)
        self.assertEqual(result["triage_urgency"], "Routine")
        self.assertIn("discharge", result["clinical_recommendation"].lower())

    def test_08_inference_engine_multiclass_diagnostic_specificity(self):
        """Verify high sensitivity and specificity across all 4 sample pathologies."""
        targets = ["Normal", "COVID19", "Bacterial_Pneumonia", "Viral_Pneumonia"]
        for target in targets:
            spath = os.path.join(self.sample_dir, f"sample_{target}.png")
            res = self.engine.predict(spath)
            self.assertEqual(res["predicted_class"], target, f"Failed prediction on {target}: got {res['predicted_class']}")
            self.assertGreaterEqual(res["confidence_score"], 0.90)

    def test_09_evaluation_metrics_thresholds(self):
        """Verify published benchmark metrics meet AlmaBetter standards (>90% accuracy)."""
        self.assertTrue(os.path.exists(self.metrics_path), "evaluation_metrics.json missing.")
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertGreaterEqual(metrics["Test_Accuracy"], 0.90, f"Accuracy too low: {metrics['Test_Accuracy']}")
        self.assertGreaterEqual(metrics["Macro_F1"], 0.90, f"Macro F1 too low: {metrics['Macro_F1']}")
        self.assertGreaterEqual(metrics["Mean_ROC_AUC"], 0.95, f"Mean ROC-AUC too low: {metrics['Mean_ROC_AUC']}")

    def test_10_submission_notebooks_completeness(self):
        """Verify submission notebook has all 6 code cells executed with output outputs."""
        self.assertTrue(os.path.exists(self.notebook_path), "Submission notebook missing.")
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 6, f"Expected 6 code cells, found {len(code_cells)}")
        for i, cell in enumerate(code_cells):
            self.assertTrue(len(cell.get("outputs", [])) > 0, f"Code cell {i+1} has empty outputs.")

if __name__ == "__main__":
    unittest.main()
