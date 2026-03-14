import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

def eval_ablation(model_name, gt_dir, pred_dir, conf_thresh=0.4, iou_thresh=0.5):
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
            area, ratio = gt_w * gt_h, gt_w / gt_h if gt_h > 0 else 1.0
            
            size_b = "Small" if area < 0.00025 else "Med" if area < 0.002 else "Large"
            vel_b = "Slow" if ratio < 1.05 else "Med" if ratio < 1.3 else "Fast"
            
            matched = False
            gt_box = [gt_x, gt_y, gt_w, gt_h]
            for p in preds:
                if calculate_iou(gt_box, p[2:6]) >= iou_thresh:
                    matched = True
                    break 
            
            stats["Size"][size_b][1] += 1
            stats["Velocity"][vel_b][1] += 1
            if matched:
                stats["Size"][size_b][0] += 1
                stats["Velocity"][vel_b][0] += 1

    row_data = {'Model': model_name}
    for cat, buckets in stats.items():
        for bucket, (tp, total) in buckets.items():
            row_data[f"{cat}_{bucket}"] = (tp / total) * 100 if total > 0 else 0
    return row_data

def plot_ablation_category(results, category, buckets, out_dir, filename, title):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    
    # Подготовка данных
    melted = []
    for _, row in df.iterrows():
        m = row['Model']
        for b in buckets:
            melted.append({'Model': m, category: b, 'Recall': row[f'{category}_{b}']})
            
    df_plot = pd.DataFrame(melted)
    
    # Отрисовка
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(9, 6))
    
    # Цвета: Базовая модель - серый, С аугментациями - синий (или красный)
    palette = {"ProxiBall (Base)": "#95a5a6", "ProxiBall (Augmented)": "#3498db"}
    
    ax = sns.barplot(data=df_plot, x=category, y='Recall', hue='Model', palette=palette, order=buckets)
    
    # Добавляем подписи со значениями над столбцами
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold', fontsize=10)
        
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel(r"Recall (%) [IoU $\geq$ 0.5]", fontweight='bold')
    plt.xlabel(f"{category} Category", fontweight='bold')
    plt.ylim(0, 110)
    plt.legend(loc='lower right')
    
    out_path = Path(out_dir) / f'Graph_Ablation_{filename}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[УСПЕХ] График абляции ({category}) сохранен в {out_path}")

if __name__ == "__main__":
    labels_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels"
    preds_root = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions"
    out_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/08_ablation"
    
    models_to_compare = {
        "ProxiBall (Base)": "ProxiBall",
        "ProxiBall (Augmented)": "ProxiBall-Augmented"
    }
    
    results = []
    for display_name, folder_name in models_to_compare.items():
        pred_dir = Path(preds_root) / folder_name
        if pred_dir.exists():
            print(f"Оценка {display_name}...")
            res = eval_ablation(display_name, labels_dir, pred_dir)
            results.append(res)
        else:
            print(f"Ожидание данных: {pred_dir} не найдена.")
            
    if len(results) > 0:
        # Plot Velocity
        plot_ablation_category(
            results, 
            category="Velocity", 
            buckets=["Slow", "Med", "Fast"], 
            out_dir=out_dir, 
            filename="Augmentations_Velocity",
            title="Ablation Study: Impact of Augmentations on Velocity-aware Recall"
        )
        
        # Plot Size
        plot_ablation_category(
            results, 
            category="Size", 
            buckets=["Small", "Med", "Large"], 
            out_dir=out_dir, 
            filename="Augmentations_Size",
            title="Ablation Study: Impact of Augmentations on Size-aware Recall"
        )