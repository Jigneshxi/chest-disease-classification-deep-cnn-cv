"""
Synthetic High-Fidelity Medical Imaging Dataset Generator for M2_DLCV
Generates 1,200 labeled chest radiological scan images across 4 distinct classes:
- Normal (Healthy lung fields)
- Bacterial_Pneumonia (Focal dense consolidation)
- Viral_Pneumonia (Diffuse interstitial patterns)
- COVID19 (Bilateral peripheral ground-glass opacities)
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSES = ["Normal", "Bacterial_Pneumonia", "Viral_Pneumonia", "COVID19"]
SPLITS = {"train": 210, "val": 45, "test": 45}  # 300 per class, 1200 total

def create_base_chest_scan(img_size=(224, 224)):
    """Creates a base thoracic radiography scan with anatomical chest outline, lungs, and spine."""
    w, h = img_size
    img = Image.new("RGB", (w, h), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)
    
    # Outer ribcage contour
    draw.ellipse([15, 20, w - 15, h - 10], fill=(35, 36, 40), outline=(50, 52, 58), width=3)
    
    # Mediastinum / Heart silhouette
    draw.polygon([(w//2 - 25, 30), (w//2 + 25, 30), (w//2 + 35, h - 35), (w//2 - 40, h - 35)], fill=(60, 62, 68))
    draw.ellipse([w//2 - 50, h//2, w//2 + 20, h - 40], fill=(70, 72, 80))  # Cardiac apex
    
    # Left and Right Lung cavities (darker radiolucent air spaces)
    draw.ellipse([28, 40, w//2 - 12, h - 45], fill=(22, 24, 28))
    draw.ellipse([w//2 + 12, 40, w - 28, h - 45], fill=(22, 24, 28))
    
    # Rib markings
    for y in range(50, h - 40, 22):
        draw.arc([22, y, w//2 - 10, y + 25], start=180, end=360, fill=(45, 47, 52), width=2)
        draw.arc([w//2 + 10, y, w - 22, y + 25], start=180, end=360, fill=(45, 47, 52), width=2)
        
    return img

def apply_pathology_features(base_img, label):
    """Applies class-specific morphological radiological signatures."""
    arr = np.array(base_img, dtype=np.float32)
    h, w, c = arr.shape
    
    # Add subtle sensor grain
    noise = np.random.normal(0, 3.5, (h, w, c))
    arr = np.clip(arr + noise, 0, 255)
    
    if label == "Normal":
        # Clear lung parenchyma with delicate bronchovascular tree
        for _ in range(8):
            lx = random.randint(35, w//2 - 25)
            ly = random.randint(60, h - 60)
            arr[ly-3:ly+3, lx-3:lx+3] += random.uniform(8, 15)
            
    elif label == "Bacterial_Pneumonia":
        # Dense localized consolidation (lobar opacity)
        side = random.choice(["left", "right"])
        center_x = random.randint(45, 75) if side == "left" else random.randint(w - 75, w - 45)
        center_y = random.randint(h//2, h - 70)
        radius = random.randint(22, 35)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        mask = np.exp(-(dist**2) / (2 * (radius**2)))
        
        for ch in range(3):
            arr[:, :, ch] += mask * random.uniform(110, 160)
            
    elif label == "Viral_Pneumonia":
        # Diffuse interstitial peribronchial thickening across both lungs
        for _ in range(30):
            cx = random.choice([random.randint(35, w//2 - 20), random.randint(w//2 + 20, w - 35)])
            cy = random.randint(55, h - 60)
            r = random.randint(6, 14)
            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = np.exp(-(dist**2) / (2 * (r**2)))
            for ch in range(3):
                arr[:, :, ch] += mask * random.uniform(45, 80)
                
    elif label == "COVID19":
        # Peripheral bilateral ground-glass opacities (GGO) along thoracic edges
        for cx, cy in [(35, random.randint(70, h - 80)), (w - 35, random.randint(70, h - 80))]:
            r = random.randint(18, 28)
            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = np.exp(-(dist**2) / (2 * (r**2)))
            for ch in range(3):
                arr[:, :, ch] += mask * random.uniform(85, 125)
                
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.7))
    return final_img

def generate_all_datasets():
    print("Generating M2_DLCV Medical Image Classification Dataset...")
    total_generated = 0
    
    for split_name, count_per_class in SPLITS.items():
        for class_name in CLASSES:
            folder = os.path.join(DATA_DIR, split_name, class_name)
            os.makedirs(folder, exist_ok=True)
            
            for idx in range(count_per_class):
                base = create_base_chest_scan()
                img = apply_pathology_features(base, class_name)
                
                filename = f"{class_name}_{split_name}_{idx+1:04d}.png"
                img.save(os.path.join(folder, filename))
                total_generated += 1
                
    # Also save representative sample images for testing
    sample_dir = os.path.join(DATA_DIR, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    for class_name in CLASSES:
        sample_img = apply_pathology_features(create_base_chest_scan(), class_name)
        sample_img.save(os.path.join(sample_dir, f"sample_{class_name}.png"))
        
    print(f"Dataset generation complete! Total images created: {total_generated} across {len(CLASSES)} classes.")

if __name__ == "__main__":
    generate_all_datasets()
