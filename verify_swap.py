import torch
from ultralytics import YOLO

def verify_swap(orig_path, swapped_path):
    print("Loading models...")
    orig_model = YOLO(orig_path)
    swapped_model = YOLO(swapped_path)
    
    # 1. Check names
    print(f"Original names[0,1]: {orig_model.names[0]}, {orig_model.names[1]}")
    print(f"Swapped names[0,1]: {swapped_model.names[0]}, {swapped_model.names[1]}")
    
    assert orig_model.names[0] == swapped_model.names[1]
    assert orig_model.names[1] == swapped_model.names[0]
    print("✓ Metadata names swap verified.")

    # 2. Check weights in first cv3 layer
    orig_detect = orig_model.model.model[-1]
    swap_detect = swapped_model.model.model[-1]
    
    orig_conv = orig_detect.cv3[0][-1] if isinstance(orig_detect.cv3[0], torch.nn.Sequential) else orig_detect.cv3[0]
    swap_conv = swap_detect.cv3[0][-1] if isinstance(swap_detect.cv3[0], torch.nn.Sequential) else swap_detect.cv3[0]
    
    with torch.no_grad():
        # Check if swapped_weights[0] == original_weights[1]
        assert torch.allclose(orig_conv.weight[0], swap_conv.weight[1])
        assert torch.allclose(orig_conv.weight[1], swap_conv.weight[0])
        
        # Check if swapped_bias[0] == original_bias[1]
        assert torch.allclose(orig_conv.bias[0], swap_conv.bias[1])
        assert torch.allclose(orig_conv.bias[1], swap_conv.bias[0])
        
    print("✓ Physics weights swap verified.")
    print("Verification SUCCESS!")

if __name__ == "__main__":
    orig = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project.pt"
    swapped = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project-swapped.pt"
    verify_swap(orig, swapped)
