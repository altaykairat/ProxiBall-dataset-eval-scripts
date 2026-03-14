import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
# ==========================================
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

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def calculate_metrics(all_preds, total_gt):
    if len(all_preds) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0 # P, R, AP
    
    all_preds.sort(key=lambda x: x[0], reverse=True)
    matches = np.array([x[1] for x in all_preds])
    confs = np.array([x[0] for x in all_preds])
    
    tps = np.cumsum(matches)
    fps = np.cumsum(1 - matches)
    
    recalls = tps / total_gt
    precisions = tps / (tps + fps + 1e-16)
    
    # Calculate AP
    ap = compute_ap(recalls, precisions)
    
    # Calculate Precision and Recall at Max F1
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
    max_f1_idx = np.argmax(f1)
    
    return precisions[max_f1_idx], recalls[max_f1_idx], ap

# ==========================================
# 2. ДВИЖОК ОЦЕНКИ
# ==========================================
def evaluate_on_dataset(model_path, img_dir, gt_dir, imgsz=960, conf=0.001):
    """Оценивает одну модель на одном датасете и возвращает набор метрик."""
    model = YOLO(model_path)
    # Используем stream=True для экономии памяти, imgsz=960 для детекции мелких объектов
    results = model(img_dir, stream=True, conf=conf, imgsz=imgsz, verbose=False)
    
    gt_paths = {p.stem: p for p in Path(gt_dir).glob('*.txt')}
    
    # Контейнеры для разных порогов IoU (от 0.5 до 0.95)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    preds_at_iou = {round(iou, 2): [] for iou in iou_thresholds}
    total_gt = 0
    
    for result in results:
        img_stem = Path(result.path).stem
        
        gts = []
        if img_stem in gt_paths:
            with open(gt_paths[img_stem], 'r') as f:
                # Фильтруем только класс мяча (обычно 0)
                gts = [list(map(float, line.strip().split())) for line in f if line.split()[0] == '0']
        total_gt += len(gts)
        
        preds_raw = []
        for box in result.boxes:
            # Берем только класс 0 (мяч)
            if int(box.cls[0].item()) == 0:
                p_conf = box.conf[0].item()
                x, y, w, h = box.xywhn[0].tolist()
                preds_raw.append([p_conf, x, y, w, h])
            
        preds_raw.sort(key=lambda x: x[0], reverse=True)
        
        # Для каждого порога IoU считаем TP/FP
        for iou_thresh in iou_thresholds:
            iou_thresh_key = round(iou_thresh, 2)
            gt_matched = [False] * len(gts)
            
            for p in preds_raw:
                p_conf = p[0]
                p_box = p[1:5]
                
                best_iou, best_gt = 0, -1
                for gt_idx, gt in enumerate(gts):
                    if not gt_matched[gt_idx]:
                        iou = calculate_iou(gt[1:5], p_box)
                        if iou > best_iou:
                            best_iou, best_gt = iou, gt_idx
                
                if best_iou >= iou_thresh:
                    gt_matched[best_gt] = True
                    preds_at_iou[iou_thresh_key].append((p_conf, 1))
                else:
                    preds_at_iou[iou_thresh_key].append((p_conf, 0))

    if total_gt == 0:
        return {"Precision": 0, "Recall": 0, "mAP50": 0, "mAP50-95": 0}

    # Метрики при IoU=0.5
    p50, r50, ap50 = calculate_metrics(preds_at_iou[0.5], total_gt)
    
    # mAP50-95
    aps = [ap50]
    for iou in iou_thresholds[1:]:
        _, _, ap_iou = calculate_metrics(preds_at_iou[round(iou, 2)], total_gt)
        aps.append(ap_iou)
    
    map50_95 = np.mean(aps)
    
    return {
        "Precision": round(p50, 4),
        "Recall": round(r50, 4),
        "mAP50": round(ap50, 4),
        "mAP50-95": round(map50_95, 4)
    }

# ==========================================
# 3. ПОСТРОЕНИЕ МАТРИЦЫ 2x2
# ==========================================
if __name__ == "__main__":
    
    # 1. СЛОВАРИ С ПУТЯМИ (Настройте под себя)
    datasets = {
        "Tested on ProxiBall": {
            "images": "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/images",
            "labels": "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels",
            "imgsz": 960,
            "conf": 0.001
        },
        "Tested on SoccerNet": {
            "images": "D:/Altay/dataset-evaluation/datasets/soccernetv3h250_blurred/valid/images", 
            "labels": "D:/Altay/dataset-evaluation/datasets/soccernetv3h250_blurred/valid/labels",
            "imgsz": 960,
            "conf": 0.0005 # Снижаем порог для Соккернет
        }
    }
    
    models = {
        "SoccerNet_Model": "D:/Altay/dataset-evaluation/6.1.data-eval/weights/soccernet.pt",
        "ProxiBall_Model": "D:/Altay/dataset-evaluation/6.1.data-eval/weights/proxiball_raw.pt"
    }
    
    out_folder = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/09_cross_dataset"
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    
    # Инициализируем словари для всех метрик
    metrics_data = {
        "Precision": [],
        "Recall": [],
        "mAP50": [],
        "mAP50-95": []
    }
    
    print("Начинаем генерацию Cross-Dataset матриц...\n")
    
    for model_name, w_path in models.items():
        if not Path(w_path).exists():
            print(f"ПРОПУСК: Веса не найдены -> {w_path}")
            continue
            
        row_res = {m: {"Model": model_name} for m in metrics_data.keys()}
        
        for dataset_name, d_config in datasets.items():
            img_dir, gt_dir = d_config["images"], d_config["labels"]
            imgsz, conf = d_config["imgsz"], d_config["conf"]
            
            if Path(img_dir).exists() and Path(gt_dir).exists():
                print(f"Оценка {model_name} на {dataset_name} (imgsz={imgsz}, conf={conf})...")
                metrics = evaluate_on_dataset(w_path, img_dir, gt_dir, imgsz=imgsz, conf=conf)
                for m_name, val in metrics.items():
                    row_res[m_name][dataset_name] = val
            else:
                print(f"ПРОПУСК: Датасет {dataset_name} не найден!")
                for m_name in metrics_data.keys():
                    row_res[m_name][dataset_name] = "N/A"
                
        for m_name in metrics_data.keys():
            metrics_data[m_name].append(row_res[m_name])

    # 4. СОХРАНЕНИЕ ТАБЛИЦ
    for m_name, data in metrics_data.items():
        if data:
            df = pd.DataFrame(data)
            csv_out = Path(out_folder) / f'Table_Cross_Dataset_{m_name}.csv'
            df.to_csv(csv_out, index=False)
            
            print(f"\n=========================================")
            print(f"МАТРИЦА: {m_name}")
            print(f"=========================================")
            print(df.to_string(index=False))
            print(f"\n[УСПЕХ] Файл сохранен: {csv_out}")