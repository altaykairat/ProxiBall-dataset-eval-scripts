import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Intersection over Union (IoU)."""
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

def extract_false_positives(model_name, model_path, img_dir, gt_dir, out_dir, imgsz=960, conf_thresh=0.4, iou_thresh=0.5, padding=120):
    out_path = Path(out_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- [Model: {model_name}] ---")
    print(f"Запуск инференса...")
    
    model = YOLO(model_path)
    results = model(img_dir, stream=True, conf=conf_thresh, imgsz=imgsz, verbose=False)
    
    gt_paths = {p.stem: p for p in Path(gt_dir).glob('*.txt')}
    
    fp_count = 0
    print(f"Поиск False Positives...")

    for result in results:
        img_path = Path(result.path)
        img_stem = img_path.stem
        
        # Читаем предсказания
        preds = []
        for box in result.boxes:
            if int(box.cls[0].item()) == 0: # Только мяч
                p_conf = box.conf[0].item()
                x, y, w, h = box.xywhn[0].tolist()
                preds.append([p_conf, x, y, w, h])
                
        if not preds:
            continue
            
        # Читаем Ground Truth
        gts = []
        if img_stem in gt_paths:
            with open(gt_paths[img_stem], 'r') as f:
                gts = [list(map(float, line.strip().split())) for line in f if line.split()[0] == '0']
                
        img = None
        
        for i, p in enumerate(preds):
            conf = p[0]
            p_box = p[1:5]
            
            best_iou = 0
            for gt in gts:
                iou = calculate_iou(gt[1:5], p_box)
                if iou > best_iou:
                    best_iou = iou
                    
            if best_iou < iou_thresh:
                if img is None:
                    img = cv2.imread(str(img_path))
                    if img is None: continue
                    img_h, img_w = img.shape[:2]
                
                x_c, y_c, w_n, h_n = p_box
                
                x_center, y_center = int(x_c * img_w), int(y_c * img_h)
                box_w, box_h = int(w_n * img_w), int(h_n * img_h)
                
                # BBox coordinates for the original image
                bx1 = x_center - box_w//2
                by1 = y_center - box_h//2
                bx2 = x_center + box_w//2
                by2 = y_center + box_h//2

                # Crop coordinates with padding
                x1 = max(0, bx1 - padding)
                y1 = max(0, by1 - padding)
                x2 = min(img_w, bx2 + padding)
                y2 = min(img_h, by2 + padding)
                
                fp_crop = img[y1:y2, x1:x2].copy()
                
                # Draw bbox on the crop (relativize coordinates to crop)
                cv2.rectangle(fp_crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (0, 0, 255), 2)
                cv2.putText(fp_crop, f"{conf:.2f}", (bx1 - x1, by1 - y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                out_name = f"{model_name}_FP_conf{conf:.2f}_{img_stem}_box{i}.jpg"
                cv2.imwrite(str(out_path / out_name), fp_crop)
                fp_count += 1

    print(f"[УСПЕХ] Найдено и сохранено {fp_count} FP для {model_name}.")

if __name__ == "__main__":
    img_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/images"
    gt_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/labels"
    
    # Мы исследуем ошибки базовой модели (например, SoccerNet)
    model_name = "ISSIA"
    model_path = "D:/Altay/dataset-evaluation/6.1.data-eval/weights/issia.pt"
    
    out_dir = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/11_false_positives_issia"
    
    extract_false_positives(
        model_name=model_name, 
        model_path=model_path, 
        img_dir=img_dir, 
        gt_dir=gt_dir, 
        out_dir=out_dir, 
        conf_thresh=0.4
    )