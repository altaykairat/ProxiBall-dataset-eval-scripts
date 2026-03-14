import torch
from ultralytics import YOLO
import os


def swap_yolo_classes(model_path, save_path, idx1=0, idx2=1):
    """
    Swaps the weights and names for two classes in a YOLO model.
    """
    print(f"Loading model from {model_path}...")
    # Load model
    model = YOLO(model_path)
    
    # 1. Swap names in metadata
    names = model.names.copy()
    print(f"Original names: {names}")
    names[idx1], names[idx2] = names[idx2], names[idx1]
    
    # Update underlying model names
    if hasattr(model.model, 'names'):
        model.model.names = names
    
    print(f"Swapped names: {model.names}")

    # 2. Swap weights and biases in classification layers (cv3)
    # In YOLO11, the Detect head has cv3 which contains the classification convolutions.
    # We need to swap the filters corresponding to the classes.
    
    # Access the underlying torch model
    m = model.model.model[-1] # Detect head
    
    if not hasattr(m, 'cv3'):
        raise AttributeError("The model head does not have 'cv3' layers. Is this a standard YOLO model?")

    print("Swapping weights in Detect head classification layers (cv3)...")
    for cv in m.cv3:
        # cv is typically a Sequential or a Conv2d
        # If it's a Sequential, the last element is the Conv2d
        conv = cv[-1] if isinstance(cv, torch.nn.Sequential) else cv
        
        with torch.no_grad():
            # Weights shape: [out_channels, in_channels, k, k]
            # Classification channels are usually at the beginning or middle depending on task
            # For Detection head: out_channels = nc + reg_max * 4 (usually)
            # Actually, in YOLO11 Detect head, cv3 predicts classes.
            # cv3[i] has c2 channels where c2 is the number of classes (if not using DFL/others in that specific layer)
            # Wait, let's verify exact indices.
            
            # Swap weights
            temp_weight = conv.weight[idx1].clone()
            conv.weight[idx1] = conv.weight[idx2]
            conv.weight[idx2] = temp_weight
            
            # Swap biases
            if conv.bias is not None:
                temp_bias = conv.bias[idx1].clone()
                conv.bias[idx1] = conv.bias[idx2]
                conv.bias[idx2] = temp_bias

    # 3. Save the modified model
    print(f"Saving modified model to {save_path}...")
    model.save(save_path)
    print("Done!")



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
    input_model = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project.pt"
    output_model = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project-swapped.pt"
    
    swap_yolo_classes(input_model, output_model, 0, 1)

    orig = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project.pt"
    swapped = "/home/altay/Desktop/Footbonaut/6.1.data-eval/weights/test-project-swapped.pt"
    verify_swap(orig, swapped)

