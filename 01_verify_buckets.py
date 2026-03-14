import cv2
import random
import os
from pathlib import Path
from collections import defaultdict

def verify_thresholds(img_dir, label_dir, out_dir, samples_per_class=10):
    """
    Randomly samples images for each size and velocity bucket for manual verification.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    
    # Adjustable Thresholds
    SIZE_THRESH = {'small': 0.00025, 'medium': 0.002} 
    VEL_THRESH = {'slow': 1.05, 'medium': 1.3}
    
    # Buckets to store image paths
    # Key: bucket_name, Value: list of (img_path, label_path, annotations)
    buckets = defaultdict(list)
    
    print(f"Scanning {len(img_paths)} images for categorization...")
    
    for img_path in img_paths:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, 'r') as f:
            annotations = []
            img_categories = set()
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls, x_c, y_c, w, h = map(float, parts[:5])
                
                # Size classification
                area = w * h
                size_cls = "Small" if area < SIZE_THRESH['small'] else "Med" if area < SIZE_THRESH['medium'] else "Large"
                
                # Velocity classification (Aspect Ratio Proxy)
                ratio = w / h if h > 0 else 1.0
                vel_cls = "Slow" if ratio < VEL_THRESH['slow'] else "Med" if ratio < VEL_THRESH['medium'] else "Fast"
                
                img_categories.add(f"Size_{size_cls}")
                img_categories.add(f"Velocity_{vel_cls}")
                
                annotations.append({
                    'box': (x_c, y_c, w, h),
                    'size': size_cls,
                    'vel': vel_cls,
                    'ratio': ratio
                })
            
            if annotations:
                for cat in img_categories:
                    buckets[cat].append((img_path, annotations))

    print("\nBucket Distribution (Total occurrences):")
    for cat in sorted(buckets.keys()):
        print(f"  {cat}: {len(buckets[cat])} images")

    # Sampling and Saving
    print(f"\nSampling up to {samples_per_class} images per class...")
    
    saved_count = 0
    for cat in sorted(buckets.keys()):
        sample_size = min(len(buckets[cat]), samples_per_class)
        samples = random.sample(buckets[cat], sample_size)
        
        cat_dir = Path(out_dir) / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, ann_list in samples:
            img = cv2.imread(str(img_path))
            if img is None: continue
            img_h, img_w = img.shape[:2]
            
            for ann in ann_list:
                x_c, y_c, w, h = ann['box']
                size_cls = ann['size']
                vel_cls = ann['vel']
                ratio = ann['ratio']
                
                # User request: Always red, text bolder
                rect_color = (0, 0, 255)  # Red BGR
                text_color = (0, 0, 255)  # Red BGR
                text_thickness = 2        # Bolder
                
                x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, 2)
                
                label_text = f"S:{size_cls} V:{vel_cls} (R:{ratio:.2f})"
                cv2.putText(img, label_text, (x1, max(y1 - 15, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, text_thickness)
            
            out_file = cat_dir / img_path.name
            cv2.imwrite(str(out_file), img)
            saved_count += 1

    print(f"\n[SUCCESS] Saved {saved_count} sample images to {out_dir}")

if __name__ == "__main__":
    # Project relative paths for Windows
    BASE_DIR = "D:/Altay/dataset-evaluation/6.1.data-eval"
    IMG_DIR = os.path.join(BASE_DIR, "testbench/testbench/test/images")
    LBL_DIR = os.path.join(BASE_DIR, "testbench/testbench/test/labels")
    OUT_DIR = os.path.join(BASE_DIR, "outputs/01_verification_stratified")
    
    verify_thresholds(IMG_DIR, LBL_DIR, OUT_DIR, samples_per_class=10)

    #Bucket Distribution (Total occurrences):
    #Size_Large: 17 images
    #Size_Med: 692 images
    #Size_Small: 294 images
    #Velocity_Fast: 147 images
    #Velocity_Med: 630 images
    #Velocity_Slow: 226 images