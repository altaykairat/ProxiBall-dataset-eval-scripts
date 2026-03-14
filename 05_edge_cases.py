import cv2
import os
from pathlib import Path

def extract_visual_proof(img_dir, custom_pred_dir, baseline_pred_dir, gt_dir, out_dir, conf_thresh=0.4):
    """
    ProxiBall vs baseline model (Soccernet)
    Draws frames (Ground Truth, ProxiBall, Baseline) for clarity.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    
    found_cases = 0
    
    print("Начинаем поиск идеальных кадров для визуального анализа...")

    for img_path in img_paths:
        txt_name = img_path.stem + ".txt"
        
        custom_path = Path(custom_pred_dir) / txt_name
        baseline_path = Path(baseline_pred_dir) / txt_name
        gt_path = Path(gt_dir) / txt_name
        
        # Skip images without a ball (Background)
        if not gt_path.exists() or os.path.getsize(gt_path) == 0:
            continue
            
        # Read predictions (filter by confidence)
        custom_preds = []
        if custom_path.exists():
            with open(custom_path, 'r') as f:
                custom_preds = [line.strip().split() for line in f if float(line.split()[1]) >= conf_thresh]
                
        baseline_preds = []
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_preds = [line.strip().split() for line in f if float(line.split()[1]) >= conf_thresh]
        
        # MAIN CONDITION: ProxiBall found the ball, and Baseline completely missed (False Negative)
        if len(custom_preds) > 0 and len(baseline_preds) == 0:
            found_cases += 1
            
            # Drawing for convenient visual selection
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            
            # Draw Ground Truth (blue)
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                    x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, "Ground Truth", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw ProxiBall prediction (green)
            for p in custom_preds:
                conf = float(p[1])
                x_c, y_c, w, h = map(float, p[2:6])
                x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"ProxiBall: {conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save the result
            out_file = Path(out_dir) / img_path.name
            cv2.imwrite(str(out_file), img)

    print(f"\n[SUCCESS] Found {found_cases} frames where ProxiBall worked and the baseline model didn't.")
    print(f"Frames saved to: {out_dir}")
    print("Просмотрите эту папку и выберите 3 лучших примера (Motion Blur, Truncated, Low Contrast) для статьи.")

if __name__ == "__main__":
    
    img_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/images"
    gt_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels"
    
    # Compare your best model with the best open-source competitor
    custom_preds = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions/ProxiBall"
    baseline_preds = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions/Soccernet"
    
    out_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/05_visual_edge_cases"
    
    extract_visual_proof(img_dir, custom_preds, baseline_preds, gt_dir, out_dir)