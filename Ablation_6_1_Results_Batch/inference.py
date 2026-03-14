deepsport = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/deepsport.pt"
dfl = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/dfl_bundesliga.pt"
issia = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/issia.pt"
old_dataset = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/old_dataset.pt"
soccernet = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/soccernet.pt"
test_proj = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project-swapped.pt"
football_ball_det = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/football-ball-det.pt"
main = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/main.pt"

testbench = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/data.yaml"


import csv
import os
from ultralytics import YOLO

def validate_model(weights_path, data_config, project_name, exp_name):
    
    model = YOLO(weights_path)

    results = model.val(
        data=data_config,    
        split='test',        
        imgsz=960,           
        batch=8,
        conf=0.001,          
        iou=0.6,             
        project=project_name, 
        name=exp_name,
        save_json=True,     
        plots=True,          
        verbose=True,
        classes = [0]
    )
    
    csv_path = os.path.join(project_name, "metrics_summary.csv")
    os.makedirs(project_name, exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Weights', 'Experiment', 'mAP50', 'mAP50-95', 'Precision', 'Recall'])
        writer.writerow([weights_path, exp_name, 
                         f"{results.box.map50:.4f}", 
                         f"{results.box.map:.4f}", 
                         f"{results.box.mp:.4f}", 
                         f"{results.box.mr:.4f}"])
    
    print(f"--- Результаты для {exp_name} ---")
    print(f"mAP@50: {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

models = {
    "deepsport": deepsport,
    "dfl": dfl,
    "issia": issia,
    "old_dataset": old_dataset,
    "soccernet": soccernet,
    "test_proj": test_proj,
    "football_ball_det": football_ball_det,
    "main": main
}

if __name__ == "__main__":
    project_name = "Ablation_6_1_Results_Batch"
    
    for name, path in models.items():
        print(f"\n{'='*50}")
        print(f"Validating model: {name}")
        print(f"Path: {path}")
        print(f"{'='*50}\n")
        
        validate_model(
            weights_path=path, 
            data_config=testbench,
            project_name=project_name,
            exp_name=f"{name}_vs_testbench"
        )