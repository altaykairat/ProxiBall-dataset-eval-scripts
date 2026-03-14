import os
import re
import cv2
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Paths
weights_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights"
testbench_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test"
images_dir = os.path.join(testbench_dir, "images")
labels_dir = os.path.join(testbench_dir, "labels")
output_dir = "/home/altay/Desktop/Footbonaut/Ablation_6_1_Results_Batch"
cm_dir = os.path.join(output_dir, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

models_paths = {
    "deepsport": os.path.join(weights_dir, "deepsport.pt"),
    "dfl": os.path.join(weights_dir, "dfl_bundesliga.pt"),
    "issia": os.path.join(weights_dir, "issia.pt"),
    "old_dataset": os.path.join(weights_dir, "old_dataset.pt"),
    "soccernet": os.path.join(weights_dir, "soccernet.pt"),
    "test_proj": os.path.join(weights_dir, "test-project-swapped.pt"),
    "football_ball_det": os.path.join(weights_dir, "football-ball-det.pt"),
    "main": os.path.join(weights_dir, "main.pt")
}

DATA_CONFIG = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/data.yaml"

IMG_SIZE = 960

# Bucketing definitions
V_BUCKETS = [0, 5, 15, 1000]
V_LABELS = ['Slow', 'Medium', 'Fast']
S_BUCKETS = [0, 400, 900, 100000]
S_LABELS = ['Small', 'Medium', 'Large']

def parse_filename(filename):
    # Matches patterns like IMG_0739_MOV-0000 or Screencast-from-2026-03-13-15-47-18_mp4-0086
    # Pattern: anything before the last dash is the sequence ID, anything after the last dash (until the first dot) is the frame index
    core_name = filename.split('.rf.')[0]
    if '-' in core_name:
        parts = core_name.split('-')
        frame_part = parts[-1].split('_')[0] # Remove _jpg if present
        if frame_part.isdigit():
            seq_id = '-'.join(parts[:-1])
            return seq_id, int(frame_part)
        # Fallback for patterns where frame index might be followed by other suffixes
        # or if the dash is not the frame separator
        for part in reversed(parts):
            clean_part = part.split('_')[0]
            if clean_part.isdigit():
                frame_part = clean_part
                seq_id = core_name.replace(f"-{part}", "")
                return seq_id, int(frame_part)
                
    return core_name, 0 # Fallback: treat whole name as seq, frame 0

def get_labels():
    labels = {}
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    print(f"Found {len(label_files)} label files.")
    
    for filename in label_files:
        path = os.path.join(labels_dir, filename)
        
        # Roboflow adds extra hashes to filenames in images_dir
        # We'll match by the core sequence ID and frame
        seq_id, frame_idx = parse_filename(filename)
            
        with open(path, 'r') as f:
            lines = f.readlines()
            boxes = []
            for line in lines:
                parts = line.split()
                if not parts: continue
                cls, x, y, w, h = map(float, parts)
                if cls == 0: # Ball
                    boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
            
            labels[filename] = {
                'seq_id': seq_id,
                'frame_idx': frame_idx,
                'boxes': boxes # Empty list if no ball
            }
    return labels

def calculate_nwd(box1, box2, constant=12.0):
    """Normalized Wasserstein Distance."""
    # box: [cx, cy, w, h] normalized
    # Scale to pixels
    b1 = [box1['x']*IMG_SIZE, box1['y']*IMG_SIZE, box1['w']*IMG_SIZE, box1['h']*IMG_SIZE]
    b2 = [box2['x']*IMG_SIZE, box2['y']*IMG_SIZE, box2['w']*IMG_SIZE, box2['h']*IMG_SIZE]
    
    dist_sq = (b1[0]-b2[0])**2 + (b1[1]-b2[1])**2 + ((b1[2]-b2[2])**2 + (b1[3]-b2[3])**2) / 4.0
    return np.exp(-np.sqrt(dist_sq) / constant)

def evaluate_model(name, path, labels_dict):
    print(f"\nEvaluating {name}...")
    model = YOLO(path)
    
    # Map label files to image files (considering Roboflow hashes)
    image_files = os.listdir(images_dir)
    label_to_img = {}
    for l_file in labels_dict:
        core_name = l_file.split('.rf.')[0]
        # Find image that starts with this core name
        for i_file in image_files:
            if i_file.startswith(core_name):
                label_to_img[l_file] = i_file
                break
    
    # Store results
    all_matches = [] # List of (gt_box, pred_box or None, metadata)
    
    for l_file, metadata in tqdm(labels_dict.items()):
        img_file = label_to_img.get(l_file)
        if not img_file:
            continue
            
        img_path = os.path.join(images_dir, img_file)
        results = model.predict(img_path, imgsz=IMG_SIZE, conf=0.1, verbose=False)[0]
        
        gt_boxes = metadata['boxes']
        pred_boxes = []
        for box in results.boxes:
            if int(box.cls[0].item()) == 0: # Ball
                # Convert to nx, ny, nw, nh
                xywh = box.xywhn[0].tolist()
                pred_boxes.append({'x': xywh[0], 'y': xywh[1], 'w': xywh[2], 'h': xywh[3], 'conf': box.conf[0].item()})

        # Match GT and Pred
        # If no GT, we still want to keep the metadata for negative sample counts if needed
        if not gt_boxes:
            all_matches.append({
                'gt': None,
                'pred': pred_boxes[0] if pred_boxes else None, # Record FP if any
                'metadata': metadata,
                'l_file': l_file,
                'nwd': 0.0
            })
            continue

        for gt in gt_boxes:
            best_match = None
            min_dist = 0.05 # Threshold in normalized coordinates
            
            for pred in pred_boxes:
                dist = np.sqrt((gt['x']-pred['x'])**2 + (gt['y']-pred['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = pred
            
            nwd_val = calculate_nwd(gt, best_match) if best_match else 0.0
            
            all_matches.append({
                'gt': gt,
                'pred': best_match,
                'metadata': metadata,
                'l_file': l_file,
                'nwd': nwd_val
            })

    return all_matches

def get_standard_metrics(name, path):
    print(f"Running standard validation for {name}...")
    model = YOLO(path)
    # val directory for CM
    val_proj = os.path.join(output_dir, "val")
    val_name = f"val_{name}"
    
    results = model.val(
        data=DATA_CONFIG,
        split='test',
        imgsz=IMG_SIZE,
        batch=8,
        conf=0.1, # Match the predict conf
        iou=0.6,
        project=val_proj,
        name=val_name,
        plots=True,
        verbose=False,
        save=False,
        classes=[0]
    )
    
    # Extract metrics
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'F1_Peak': np.max(results.box.f1) if hasattr(results.box, 'f1') and len(results.box.f1) > 0 else 0.0
    }
    
    # Find and copy confusion matrix
    cm_path = os.path.join(val_proj, val_name, "confusion_matrix.png")
    if os.path.exists(cm_path):
        target_cm = os.path.join(cm_dir, f"CM_{name}.png")
        shutil.copy(cm_path, target_cm)
        metrics['cm_path'] = target_cm
    
    return metrics

def post_process_metrics(name, all_matches, std_metrics):
    # Sort matches to calculate velocity
    rows = []
    for m in all_matches:
        if m['gt']:
            rows.append({
                'l_file': m['l_file'],
                'seq_id': m['metadata']['seq_id'],
                'frame_idx': m['metadata']['frame_idx'],
                'gt_x': m['gt']['x'],
                'gt_y': m['gt']['y'],
                'gt_w': m['gt']['w'],
                'gt_h': m['gt']['h'],
                'pred_x': m['pred']['x'] if m['pred'] else None,
                'pred_y': m['pred']['y'] if m['pred'] else None,
                'detected': 1 if m['pred'] else 0,
                'area': m['gt']['w'] * m['gt']['h'] * (IMG_SIZE**2),
                'nwd': m['nwd']
            })
        else:
            # Image with no ball
            rows.append({
                'l_file': m['l_file'],
                'seq_id': m['metadata']['seq_id'],
                'frame_idx': m['metadata']['frame_idx'],
                'gt_x': None,
                'gt_y': None,
                'gt_w': None,
                'gt_h': None,
                'pred_x': m['pred']['x'] if m['pred'] else None,
                'pred_y': m['pred']['y'] if m['pred'] else None,
                'detected': 0, # Not a recall-related detection
                'area': 0,
                'nwd': 0.0
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['seq_id', 'frame_idx'])
    
    # Velocity calculation (displacement in pixels)
    df['velocity'] = 0.0
    for seq_id, group in df.groupby('seq_id'):
        group_gt = group[group['gt_x'].notna()]
        if len(group_gt) > 1:
            coords = group_gt[['gt_x', 'gt_y']].values * IMG_SIZE
            diffs = np.diff(coords, axis=0)
            dists = np.sqrt((diffs**2).sum(axis=1))
            # Prepend 0 for the first frame of sequence
            df.loc[group_gt.index[1:], 'velocity'] = dists
            # For the first frame, use the velocity of the second frame as proxy
            df.loc[group_gt.index[0], 'velocity'] = dists[0]

    # RMSE for detected boxes
    detected_df = df[(df['gt_x'].notna()) & (df['pred_x'].notna())]
    if not detected_df.empty:
        rmse = np.sqrt(np.mean(((detected_df['gt_x'] - detected_df['pred_x'])*IMG_SIZE)**2 + 
                               ((detected_df['gt_y'] - detected_df['pred_y'])*IMG_SIZE)**2))
        avg_nwd = detected_df['nwd'].mean()
    else:
        rmse = 0.0
        avg_nwd = 0.0
        
    # Bucketing
    # Only for images with GT
    gt_df = df[df['gt_x'].notna()].copy()
    if not gt_df.empty:
        gt_df['v_bucket'] = pd.cut(gt_df['velocity'], bins=V_BUCKETS, labels=V_LABELS)
        gt_df['s_bucket'] = pd.cut(gt_df['area'], bins=S_BUCKETS, labels=S_LABELS)
        
        v_recall = gt_df.groupby('v_bucket')['detected'].mean()
        s_recall = gt_df.groupby('s_bucket')['detected'].mean()
        total_recall = gt_df['detected'].mean()
    else:
        v_recall = pd.Series()
        s_recall = pd.Series()
        total_recall = 0.0
    
    return {
        'RMSE': rmse,
        'NWD': avg_nwd,
        'v_recall': v_recall.to_dict(),
        's_recall': s_recall.to_dict(),
        'total_recall': total_recall,
        'std': std_metrics
    }

if __name__ == "__main__":
    labels = get_labels()
    print(f"Total labels loaded after parsing: {len(labels)}")
    
    results_summary = []
    
    all_metrics = {}
    
    for name, path in models_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, path not found: {path}")
            continue
            
        std_metrics = get_standard_metrics(name, path)
        matches = evaluate_model(name, path, labels)
        metrics = post_process_metrics(name, matches, std_metrics)
        all_metrics[name] = metrics
        
        results_summary.append({
            'Model': name,
            'Precision': f"{metrics['std']['Precision']:.4f}",
            'Recall': f"{metrics['std']['Recall']:.4f}",
            'mAP50': f"{metrics['std']['mAP50']:.4f}",
            'mAP50-95': f"{metrics['std']['mAP50-95']:.4f}",
            'NWD': f"{metrics['NWD']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}",
            'Slow Recall': f"{metrics['v_recall'].get('Slow', 0):.4f}",
            'Med Recall': f"{metrics['v_recall'].get('Medium', 0):.4f}",
            'Fast Recall': f"{metrics['v_recall'].get('Fast', 0):.4f}",
            'Small Recall': f"{metrics['s_recall'].get('Small', 0):.4f}",
            'Large Recall': f"{metrics['s_recall'].get('Large', 0):.4f}"
        })

    # Save CSV (Updating metrics_summary.csv as requested)
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print(f"\nResults saved to {output_dir}/metrics_summary.csv")
    
    # Plotting Recall vs Velocity
    plt.figure(figsize=(10, 6))
    models = list(all_metrics.keys())
    x = np.arange(len(V_LABELS))
    width = 0.1
    
    for i, model in enumerate(models):
        recalls = [all_metrics[model]['v_recall'].get(l, 0) for l in V_LABELS]
        plt.bar(x + i*width - (len(models)*width)/2, recalls, width, label=model)
    
    plt.xlabel('Velocity Bucket')
    plt.ylabel('Recall')
    plt.title('Recall vs Ball Velocity')
    plt.xticks(x, V_LABELS)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "recall_vs_velocity.png"))
    
    # Plotting Recall vs Size
    plt.figure(figsize=(10, 6))
    x = np.arange(len(S_LABELS))
    for i, model in enumerate(models):
        recalls = [all_metrics[model]['s_recall'].get(l, 0) for l in S_LABELS]
        plt.bar(x + i*width - (len(models)*width)/2, recalls, width, label=model)
        
    plt.xlabel('Ball Size Bucket (px^2)')
    plt.ylabel('Recall')
    plt.title('Recall vs Ball Size')
    plt.xticks(x, S_LABELS)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "recall_vs_size.png"))

    # Combined Confusion Matrix Tiling
    print("Generating combined confusion matrix...")
    cm_files = [os.path.join(cm_dir, f"CM_{name}.png") for name in models if os.path.exists(os.path.join(cm_dir, f"CM_{name}.png"))]
    if cm_files:
        n = len(cm_files)
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*6))
        axes = axes.flatten()
        
        for i, cm_file in enumerate(cm_files):
            model_name = os.path.basename(cm_file).replace("CM_", "").replace(".png", "")
            img = plt.imread(cm_file)
            axes[i].imshow(img)
            axes[i].set_title(f"Confusion Matrix: {model_name}")
            axes[i].axis('off')
            
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_confusion_matrix.png"))
        print(f"Combined CM saved to {output_dir}/combined_confusion_matrix.png")

