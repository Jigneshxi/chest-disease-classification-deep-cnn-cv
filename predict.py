"""
Production Inference Engine & Clinical Triage Classifier: M2_DLCV Computer Vision
AlmaBetter AI Engineer Track (AB_AIE) - Module 2: Deep Learning for Computer Vision (M2_DLCV)
"""

import os
import sys
import json
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

class ChestXRayCNN(nn.Module):
    """4-Stage Deep Convolutional Neural Network for Thoracic Radiography."""
    def __init__(self, num_classes=4):
        super(ChestXRayCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))

class CVInferenceEngine:
    """Production Computer Vision Inference and Triage Classifier."""
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "models")
            
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "best_cv_model.pt")
        self.classes_path = os.path.join(model_dir, "class_names.json")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()
        
        self.eval_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(f"Class names list not found: {self.classes_path}")
            
        with open(self.classes_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
            
        self.model = ChestXRayCNN(num_classes=len(self.class_names)).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError("Unsupported image type. Provide file path, PIL.Image, or numpy.ndarray.")
            
        tensor = self.eval_transforms(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input):
        tensor = self.preprocess_image(image_input)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        confidence = float(probs[top_idx])
        
        prob_dist = {cname: round(float(probs[i]), 4) for i, cname in enumerate(self.class_names)}
        
        # Clinical Triage Stratification
        triage_recommendations = {
            "Normal": {
                "triage_urgency": "Routine",
                "finding": "Clear bilateral lung parenchyma without acute focal consolidation.",
                "action": "Routine preventive healthcare maintenance; discharge."
            },
            "Bacterial_Pneumonia": {
                "triage_urgency": "High Priority",
                "finding": "Dense lobar opacity consistent with bacterial alveolar consolidation.",
                "action": "Urgent physician consult, prescribe targeted antibiotic therapy, sputum culture."
            },
            "Viral_Pneumonia": {
                "triage_urgency": "High Priority",
                "finding": "Diffuse bilateral interstitial infiltrates and peribronchial thickening.",
                "action": "Supportive respiratory therapy, viral antigen panel, monitor SpO2 saturation."
            },
            "COVID19": {
                "triage_urgency": "Critical Isolation",
                "finding": "Peripheral bilateral ground-glass opacities (GGO) with subpleural distribution.",
                "action": "Immediate negative-pressure respiratory isolation, stat RT-PCR testing, prone positioning."
            }
        }
        
        triage = triage_recommendations.get(top_class, {
            "triage_urgency": "Standard",
            "finding": "Anomalous thoracic radiographic pattern.",
            "action": "Refer to radiologist for manual second-read."
        })
        
        return {
            "predicted_class": top_class,
            "confidence_score": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "class_probabilities": prob_dist,
            "triage_urgency": triage["triage_urgency"],
            "radiological_finding": triage["finding"],
            "clinical_recommendation": triage["action"]
        }

def main():
    parser = argparse.ArgumentParser(description="M2_DLCV Computer Vision Inference CLI")
    parser.add_argument("--image", type=str, help="Path to chest x-ray image file.")
    parser.add_argument("--batch_dir", type=str, help="Directory containing images for batch prediction.")
    parser.add_argument("--output", type=str, help="Optional output JSON path.")
    args = parser.parse_args()

    engine = CVInferenceEngine()
    
    if args.image:
        res = engine.predict(args.image)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Prediction saved to {args.output}")
        else:
            print(json.dumps(res, indent=2))
    elif args.batch_dir:
        files = [os.path.join(args.batch_dir, f) for f in os.listdir(args.batch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        batch_results = []
        for fpath in files:
            r = engine.predict(fpath)
            r["filename"] = os.path.basename(fpath)
            batch_results.append(r)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, indent=2)
            print(f"Batch predictions for {len(batch_results)} images saved to {args.output}")
        else:
            print(json.dumps(batch_results, indent=2))
    else:
        # Default self-test using sample normal image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, "data", "sample_images", "sample_Normal.png")
        if os.path.exists(sample_path):
            res = engine.predict(sample_path)
            print("Self-Test Inference Output (Sample Normal Image):")
            print(json.dumps(res, indent=2))
        else:
            print("Please specify an image with --image <path>")

if __name__ == "__main__":
    main()
