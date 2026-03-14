import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

def generate_rmse_and_cm(models, gt_dir, preds_root_dir, out_dir, conf_thresh=0.4, iou_thresh=0.5):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gt_paths = list(Path(gt_dir).glob('*.txt'))
    
    img_w, img_h = 960, 960 # Corrected resolution
    rmse_stratified = []
    
    print("Starting RMSE calculation and Confusion Matrix generation...\n")

    for model_name in models:
        pred_dir = Path(preds_root_dir) / model_name
        if not pred_dir.exists():
            continue
            
        tp, fp, fn = 0, 0, 0
        sum_sq_error_global = 0.0
        
        # Reservoirs for stratified RMSE
        # {Category: {Bucket: [sum_sq_error, count]}}
        strat_stats = {
            "Size": {"Small": [0.0, 0], "Med": [0.0, 0], "Large": [0.0, 0]},
            "Velocity": {"Slow": [0.0, 0], "Med": [0.0, 0], "Fast": [0.0, 0]}
        }
        
        for gt_path in gt_paths:
            pred_path = pred_dir / gt_path.name
            
            with open(gt_path, 'r') as f:
                gts = [list(map(float, line.strip().split())) for line in f]
            
            preds = []
            if pred_path.exists():
                with open(pred_path, 'r') as f:
                    preds = [list(map(float, line.strip().split())) for line in f if float(line.split()[1]) >= conf_thresh]
            
            preds.sort(key=lambda x: x[1], reverse=True)
            gt_matched = [False] * len(gts)
            
            for p in preds:
                best_iou, best_gt_idx = 0, -1
                for idx, gt in enumerate(gts):
                    if not gt_matched[idx]:
                        iou = calculate_iou(gt[1:5], p[2:6])
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, idx
                
                if best_iou >= iou_thresh:
                    gt_matched[best_gt_idx] = True
                    tp += 1
                    
                    # Localization errors
                    gt_x, gt_y, gt_w, gt_h = gts[best_gt_idx][1:5]
                    p_x, p_y = p[2], p[3]
                    
                    err2 = (p_x * img_w - gt_x * img_w)**2 + (p_y * img_h - gt_y * img_h)**2
                    sum_sq_error_global += err2
                    
                    # CATEGORIZATION FOR STRATIFICATION
                    area = gt_w * gt_h
                    ratio = max(gt_w, gt_h) / min(gt_w, gt_h) if min(gt_w, gt_h) > 0 else 1.0
                    
                    size_b = "Small" if area < 0.00025 else "Med" if area < 0.002 else "Large"
                    vel_b = "Slow" if ratio < 1.05 else "Med" if ratio < 1.3 else "Fast"
                    
                    strat_stats["Size"][size_b][0] += err2
                    strat_stats["Size"][size_b][1] += 1
                    strat_stats["Velocity"][vel_b][0] += err2
                    strat_stats["Velocity"][vel_b][1] += 1
                else:
                    fp += 1
            
            fn += sum(1 for m in gt_matched if not m)
            
        # Final calculation
        rmse_global = np.sqrt(sum_sq_error_global / tp) if tp > 0 else 0.0
        row = {'Model': model_name, 'RMSE_Global': round(rmse_global, 2)}
        
        for category, buckets in strat_stats.items():
            for bucket, (ss_err, count) in buckets.items():
                b_rmse = np.sqrt(ss_err / count) if count > 0 else 0.0
                row[f'RMSE_{category}_{bucket}'] = round(b_rmse, 2)
        
        rmse_stratified.append(row)
        
        # Confusion Matrix Plotting
        plt.figure(figsize=(7, 6))
        cm_matrix = np.array([[tp, fn], [fp, 0]])
        
        # Use a consistent color map
        cmap = sns.light_palette("seagreen" if model_name == "ProxiBall" else "steelblue", as_cmap=True)
        
        ax = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Predicted: Ball', 'Predicted: BG'],
                    yticklabels=['Actual: Ball', 'Actual: BG'],
                    annot_kws={"size": 16, "weight": "bold"},
                    cbar=False, square=True)
        
        # Style adjustments for "premium" look
        plt.title(f'Confusion Matrix: {model_name}', fontsize=16, fontweight='bold', pad=20)
        
        # Bold borders
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')
            
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'CM_{model_name}.png', dpi=300)
        plt.close()

        print(f"[{model_name}] Global RMSE: {rmse_global:.2f} px")

    if rmse_stratified:
        csv_path = Path(out_dir) / 'Table_2_RMSE_Stratified.csv'
        pd.DataFrame(rmse_stratified).to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Table 2 (Stratified RMSE) saved to: {csv_path}")

if __name__ == "__main__":
    labels = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels"
    preds_root = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions"
    outputs = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/06_rmse_and_cm"
    
    models_to_eval = [
        "ISSIA", "YOLOv11s", "Soccernet", "DFL", "Football-Ball-Detection",
        "Ball-Detection", "Test-Project", "ProxiBall", "ProxiBall-Augmented"
    ]
    
    # Run evaluation (use threshold 0.4 to simulate real system operation)
    generate_rmse_and_cm(models_to_eval, labels, preds_root, outputs, conf_thresh=0.4)