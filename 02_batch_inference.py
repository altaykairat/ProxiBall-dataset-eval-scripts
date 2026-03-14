import gc
import torch
from pathlib import Path
from ultralytics import YOLO

def run_batch_inference(models_dict, img_dir, out_root, conf_thresh=0.001, chunk_size=8):
    
    # Runs YOLO inference and saves predictions for metric evaluation with memory optimization
    img_dir_path = Path(img_dir)

    # Get all image paths. List of strings/Path is small in RAM.
    img_paths = sorted(list(img_dir_path.glob('*.jpg')) + list(img_dir_path.glob('*.png')))
    print(f"Total images found: {len(img_paths)}")
    
    for model_name, weights_path in models_dict.items():
        print(f"\n--- Running Inference: {model_name} ---")
        try:
            model = YOLO(weights_path)
            out_dir = Path(out_root) / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Process images in small chunks to avoid VRAM OOM during pre-processing
            for i in range(0, len(img_paths), chunk_size):
                chunk = img_paths[i:i + chunk_size]
                chunk_str = [str(p) for p in chunk] # Convert to strings for reliability
                print(f"  [{model_name}] Processing chunk {i//chunk_size + 1}/{(len(img_paths) + chunk_size - 1)//chunk_size}...")
                
                # Use zip to ensure we match the result to the correct original filename
                results = model(chunk_str, stream=True, conf=conf_thresh, verbose=False) 
                
                for img_p, result in zip(chunk, results):
                    txt_path = out_dir / f"{img_p.stem}.txt"
                    with open(txt_path, 'w') as f:
                        for box in result.boxes:
                            cls = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            x, y, w, h = box.xywhn[0].tolist()
                            f.write(f"{cls} {conf:.4f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
                # Cleanup after each chunk to be safe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Explicit cleanup after each model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            # Ensure cleanup even on error
            if 'model' in locals(): del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    models = {
        "ProxiBall-Augmented" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/proxiball_augmented.pt",
        "ProxiBall" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/proxiball_raw.pt",
        "Soccernet": "D:/Altay/dataset-evaluation/6.1.data-eval/weights/soccernet.pt",
        "DFL" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/dfl_bundesliga.pt",
        "Football-Ball-Detection" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/football-ball-det.pt",
        "ISSIA" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/issia.pt",
        "Ball-Detection" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/old_dataset.pt",
        "Test-Project" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/test-project.pt",
        "Yolo11s" : "D:/Altay/dataset-evaluation/6.1.data-eval/weights/yolo11s.pt" 
    }
    
    img_dir = 'D:/Altay/dataset-evaluation/6.1.data-eval/testbench/testbench/test/images'
    out_dir = 'D:/Altay/dataset-evaluation/6.1.data-eval/outputs/02_predictions'
    
    run_batch_inference(models, img_dir, out_dir, chunk_size=128)