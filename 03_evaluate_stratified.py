import csv
import numpy as np
from pathlib import Path

# IoU & NWD

def calculate_iou(box1, box2):
    """
    IoU
    """
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

def calculate_nwd(box1, box2, img_w=960, img_h=960, C=6.4):
    """
    NWD for micro-objects.
    """
    # normalized -> pixels
    cx1, cy1, w1, h1 = box1[0]*img_w, box1[1]*img_h, box1[2]*img_w, box1[3]*img_h
    cx2, cy2, w2, h2 = box2[0]*img_w, box2[1]*img_h, box2[2]*img_w, box2[3]*img_h
    
    # Wasserstein Distance (W_2^2)
    center_dist2 = (cx1 - cx2)**2 + (cy1 - cy2)**2
    shape_dist2 = ((w1 - w2)**2 + (h1 - h2)**2) / 4.0
    w2_squared = center_dist2 + shape_dist2
    
    # Normalization
    nwd = np.exp(-np.sqrt(w2_squared) / C)
    return nwd


def calculate_stratified_recall(model_name, gt_dir, pred_dir, conf_thresh, match_metric, match_thresh):
    """
    Recall with size and velocity stratification.
    """
    gt_paths = list(Path(gt_dir).glob('*.txt'))
    
    stats = {
        "Size": {"Small": [0,0], "Med": [0,0], "Large": [0,0]},
        "Velocity": {"Slow": [0,0], "Med": [0,0], "Fast": [0,0]}
    }
    
    for gt_path in gt_paths:
        pred_path = Path(pred_dir) / gt_path.name
        
        with open(gt_path, 'r') as f:
            gts = [list(map(float, line.strip().split())) for line in f]
            
        preds = []
        if pred_path.exists():
            with open(pred_path, 'r') as f:
                preds = [list(map(float, line.strip().split())) for line in f if float(line.split()[1]) >= conf_thresh]
        
        for gt in gts:
            _, gt_x, gt_y, gt_w, gt_h = gt
            area = gt_w * gt_h
            ratio = gt_w / gt_h if gt_h > 0 else 1.0
            
            size_b = "Small" if area < 0.00025 else "Med" if area < 0.002 else "Large"
            vel_b = "Slow" if ratio < 1.05 else "Med" if ratio < 1.3 else "Fast"
            
            # Matching metric selection
            matched = False
            gt_box = [gt_x, gt_y, gt_w, gt_h]
            
            for p in preds:
                pred_box = p[2:6] 
                
                if match_metric == 'iou':
                    score = calculate_iou(gt_box, pred_box)
                elif match_metric == 'nwd':
                    score = calculate_nwd(gt_box, pred_box)
                else:
                    raise ValueError("Metric must be 'iou' or 'nwd'")
                
                if score >= match_thresh:
                    matched = True
                    break 
            
            # (Index 0: TP, Index 1: Total GT)
            stats["Size"][size_b][1] += 1
            stats["Velocity"][vel_b][1] += 1
            if matched:
                stats["Size"][size_b][0] += 1
                stats["Velocity"][vel_b][0] += 1

    row_data = {'Model': model_name}
    
    print(f"\n--- {model_name} | conf>={conf_thresh} | {match_metric.upper()}>={match_thresh} ---")
    for category, buckets in stats.items():
        for bucket, (tp, total) in buckets.items():
            recall = (tp / total) * 100 if total > 0 else 0
            print(f"{category} - {bucket}: {recall:.1f}% ({tp}/{total})")
            
            row_data[f"{category}_{bucket}_Recall(%)"] = round(recall, 2)
            row_data[f"{category}_{bucket}_TP"] = tp
            row_data[f"{category}_{bucket}_Total"] = total

    return row_data


if __name__ == "__main__":
    
    labels_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels"
    
    models = [
        "Soccernet", "DFL", 
        "Football-Ball-Detection", "ISSIA", 
        "Ball-Detection", "Test-Project", 
        "ProxiBall", "ProxiBall-Augmented", "Yolo11s"
    ]
    
    CONF_THRESH = 0.4          
    MATCH_METRIC = 'iou'    # nwd or iou   
    MATCH_THRESH = 0.5         
    
    all_results = []
    
    for model_name in models:
        pred_dir = f'D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions/{model_name}'
        
        if not Path(pred_dir).exists():
            print(f"Пропущен {model_name}: нет предсказаний")
            continue
            
        row = calculate_stratified_recall(model_name, labels_dir, pred_dir, 
                                          conf_thresh=CONF_THRESH, 
                                          match_metric=MATCH_METRIC, 
                                          match_thresh=MATCH_THRESH)
        all_results.append(row)

    if all_results:
        # Dynamic filename to avoid overwriting test results
        out_csv = f"D:/Altay/dataset-evaluation/6.1.data-eval/outputs/03_stratified/03_stratified_{MATCH_METRIC}_{MATCH_THRESH}.csv"
        
        headers = all_results[0].keys()
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)
            
        print(f"\n[УСПЕХ] Сохранено в: {out_csv}")