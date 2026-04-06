# ProxiBall Dataset Ablation Study - Evaluation Scripts

## Overview

This repository contains a comprehensive suite of **11 Python evaluation scripts** designed to conduct rigorous ablation studies and comparative analysis of the ProxiBall dataset. These scripts facilitate detailed performance evaluation of ball detection models trained on close-proximity, high-speed indoor football scenarios (Footbonaut arena).

The core objective is to quantify the performance difference between models trained on traditional broadcast-view datasets versus the custom ProxiBall dataset in challenging indoor training environments.

## Problem Statement

Standard object detection models trained on open-source sports datasets (SoccerNet, DFL Bundesliga, ISSIA-CNR) struggle significantly in close-proximity, high-speed indoor scenarios because:

- **Motion Blur**: High-speed ball movement (100+ km/h) creates severe motion blur, producing "ghost balls" in frames
- **Scale Variance**: Balls in close proximity appear much larger than in broadcast footage
- **Lighting Conditions**: Indoor LED arrays create static specular highlights and inconsistent illumination
- **Spatial Domain Gap**: Broadcast datasets feature stadium/field views; Footbonaut arena data has fundamentally different spatial characteristics

## Repository Structure

```
ProxiBall-dataset-eval-scripts/
├── 01_verify_buckets.py           # Data verification and stratification
├── 02_batch_inference.py           # Memory-efficient batch YOLO inference
├── 03_evaluate_stratified.py       # Stratified evaluation by size/velocity
├── 04_core_metrics.py              # Standard COCO & NWD metrics calculation
├── 05_edge_cases.py                # Visual analysis of edge cases
├── 06_rmse_confusion.py            # RMSE and confusion matrix analysis
├── 07_rmse_recall_diagrams.py      # Visualization of stratified metrics
├── 08_augmentation_ablation.py     # Ablation study on data augmentation
├── 09_cross_dataset_eval.py        # Cross-dataset generalization tests
├── 10_dataset_stat.py              # Dataset exploratory data analysis (EDA)
├── 11_false_positives.py           # False positive profiling
├── fix_labels.py                   # Utility for label corrections
├── swap_classes.py                 # Class swapping utility
├── requirements.txt                # Python dependencies
└── weights/                        # Pre-trained model weights directory
```

## Key Evaluation Scripts

### Stage 1: Data Verification & Inference
- **`01_verify_buckets.py`**: Stratifies test images into size and velocity buckets based on bounding box properties. Generates sample visualizations with labeled annotations.
- **`02_batch_inference.py`**: Runs memory-optimized YOLO inference across multiple models with configurable confidence thresholds and chunk processing.

### Stage 2: Metric Computation
- **`03_evaluate_stratified.py`**: Segments the test bench into 6 physical categories (3 size × 3 velocity buckets) and computes per-category performance metrics.
- **`04_core_metrics.py`**: Calculates standard COCO metrics (mAP50, mAP50-95) and the specialized **Normalized Wasserstein Distance (NWD-mAP)** metric for micro-objects.
- **`06_rmse_confusion.py`**: Computes Root Mean Squared Error (RMSE) for bounding box centroids and generates confusion matrices.

### Stage 3: Visual Analysis & Ablation
- **`05_edge_cases.py`**: Automatically identifies and visualizes failure cases where baseline models fail but ProxiBall succeeds (motion blur, truncation, low contrast).
- **`07_rmse_recall_diagrams.py`**: Generates scatter plots and bar charts comparing RMSE vs Recall across stratified buckets.
- **`08_augmentation_ablation.py`**: Quantifies the performance impact of individual augmentation strategies (Mosaic, Mixup, Motion Blur, HSV perturbation).
- **`09_cross_dataset_eval.py`**: Tests cross-dataset generalization by training on broadcast datasets and evaluating on ProxiBall test set.
- **`10_dataset_stat.py`**: Produces exploratory data analysis including spatial heatmaps, bounding box size distributions, and aspect ratio histograms.
- **`11_false_positives.py`**: Extracts and categorizes false positive detections to reveal what models confuse with balls (reflections, player limbs, court markings).

## Requirements

### System Requirements
- Python 3.11
- CUDA-compatible GPU (optional but recommended for inference)
- 16GB+ RAM for batch processing

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- **PyTorch** (2.5.1 with CUDA 12.4 support)
- **Ultralytics YOLOv8/v11** (for model inference)
- **OpenCV** (image processing and visualization)
- **NumPy, SciPy** (numerical computations)
- **Pandas** (data manipulation)
- **Matplotlib, Seaborn** (plotting and visualization)
- **Albumentations** (data augmentation)
- **FilterPy** (Kalman filtering for tracking)

## Dataset Format

The ProxiBall dataset uses YOLO format annotations:

```
Dataset Structure:
├── test/
   ├── images/ (1095 frames)
   └── labels/  (YOLO format: class_id x_center y_center width height)

```

**Annotation Fields** (YOLO Format):
- `class_id`: 0 (ball)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized bounding box dimensions (0-1)

### Bucket Classification
- **Size Buckets**: Small (<0.00025 area), Medium (0.00025-0.002), Large (>0.002)
- **Velocity Buckets**: Slow (w/h < 1.05), Medium (1.05-1.3), Fast (>1.3)
  - Velocity is estimated via aspect ratio as a proxy for motion blur

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/altaykairat/ProxiBall-dataset-eval-scripts.git
cd ProxiBall-dataset-eval-scripts
pip install -r requirements.txt
```

### 2. Configure Paths
Edit the `__main__` section in each script to set:
- `BASE_DIR`: Root directory of dataset
- `IMG_DIR`: Path to test images
- `LBL_DIR`: Path to test labels
- `OUT_DIR`: Output directory for results
- `models`: Dictionary mapping model names to weight paths

### 3. Run Evaluation Pipeline (Sequential)

```bash
# Step 1: Verify dataset stratification
python 01_verify_buckets.py

# Step 2: Run inference on all models
python 02_batch_inference.py

# Step 3: Compute core metrics
python 04_core_metrics.py

# Step 4: Stratified evaluation
python 03_evaluate_stratified.py

# Step 5: RMSE and confusion matrices
python 06_rmse_confusion.py

# Step 6: Visualization
python 07_rmse_recall_diagrams.py
python 05_edge_cases.py

# Step 7: Ablation studies
python 08_augmentation_ablation.py
python 09_cross_dataset_eval.py
python 10_dataset_stat.py
python 11_false_positives.py
```

### 4. View Results
Output files are organized in the `outputs/` directory:

```
outputs/
├── 01_verification_stratified/        # Sample images per bucket
├── 02_predictions/                    # Raw model predictions
├── 03_stratified_eval/                # Stratified metrics per bucket
├── 04_core_metrics/                   # COCO + NWD metrics
├── 06_rmse_and_cm/                    # RMSE values and confusion matrices
├── 07_visualizations/                 # Graphs and diagrams
├── 08_ablation/                       # Augmentation ablation results
├── 09_cross_dataset/                  # Cross-dataset evaluation
├── 10_dataset_eda/                    # Dataset statistics
└── 11_false_positives/                # False positive analysis
```

## Key Metrics Explained

### Standard Metrics
- **mAP50**: Mean Average Precision at IoU threshold 0.5 (strict localization requirement)
- **mAP50-95**: mAP averaged across IoU thresholds 0.5:0.95 (COCO standard)
- **Recall**: Percentage of ground truth objects detected (sensitivity)
- **Precision**: Percentage of detections that are correct (specificity)

### Specialized Metrics
- **NWD-mAP** (Normalized Wasserstein Distance): For micro-objects, pixel-level shifts collapse IoU. NWD treats bounding boxes as probability distributions, providing stable gradients even for 1-2 pixel localization errors.
- **RMSE** (Root Mean Squared Error): Pixel-space error in centroid localization. Critical for understanding precision in high-speed tracking scenarios.

## Evaluation Results Summary

### Testbench Distribution (1,003 Ground Truths)
| Category | Count | Percentage |
|----------|-------|-----------|
| Small Balls | 294 | 29.3% |
| Medium Balls | 692 | 69.0% |
| Large Balls | 17 | 1.7% |
| **Slow Velocity** | **226** | **22.5%** |
| **Medium Velocity** | **630** | **62.8%** |
| **Fast Velocity** | **147** | **14.7%** |

### Performance Comparison (IoU=0.5 Threshold)

| Model | mAP50 | mAP50-95 | NWD-mAP | RMSE (px) | Fast Recall | Small Recall |
|-------|-------|----------|---------|-----------|------------|--------------|
| **ProxiBall** | **0.979** | **0.662** | **0.979** | **1.80** | **98.6%** | **89.5%** |
| SoccerNet | 0.840 | 0.506 | 0.834 | 2.05 | 83.7% | 31.3% |
| Ball-Detection | 0.860 | 0.475 | 0.825 | 2.31 | 68.0% | 48.6% |
| DFL Bundesliga | 0.825 | 0.482 | 0.812 | 2.18 | 76.9% | 28.4% |

**Key Finding**: ProxiBall achieves **98.6% recall on fast-moving balls** compared to 83.7% for SoccerNet, demonstrating superior resilience to motion blur.

## Advanced Features

### Memory Optimization
- **Chunked Inference** (`02_batch_inference.py`): Processes images in configurable batch sizes to prevent GPU OOM
- **Explicit GPU Cleanup**: Clears CUDA cache after each model to handle sequential model loading

### Stratified Analysis
- **Size-based Buckets**: Isolates micro-object detection challenges
- **Velocity-based Buckets**: Quantifies motion blur resilience
- **Combined Analysis**: Reveals interaction effects (e.g., fast + small = extremely difficult)

### Visualization
- **Precision-Recall Curves**: Compares models across confidence thresholds
- **Stratified Bar Charts**: Shows performance breakdown by bucket
- **Scatter Plots**: Reveals RMSE vs Recall trade-offs
- **Confusion Matrices**: Visualizes True Positive, False Positive, False Negative distributions

## Ablation Study Methodology

All ablation studies follow a **strict curriculum learning approach**:

1. **Baseline Training**: Train on raw dataset without augmentation
2. **Progressive Addition**: Incrementally add augmentations and measure impact
3. **Fair Comparison**: Fix optimizer (AdamW), learning rate, and batch size across all experiments
4. **Statistical Validation**: Report mean ± std across multiple runs

## Citation

If you use this repository or dataset in your research, please cite:

```bibtex
@misc{ProxiBall,
    title={ProxiBall Dataset Ablation Study - Close-Proximity High-Speed Ball Detection},
    author={Altay Kairat},
    year={2026},
    publisher={GitHub},
    journal={GitHub Repository},
    url={https://github.com/altaykairat/ProxiBall-dataset-eval-scripts}
}
```

## License

This repository is licensed under the **cc-by-nc License**. See the `LICENSE` file for full details.

## Contact & Issues

For questions, issues, or contributions:
- Open an [Issue](https://github.com/altaykairat/ProxiBall-dataset-eval-scripts/issues) on GitHub
- Provide detailed error messages and reproduction steps
- Include environment details (Python version, PyTorch version, GPU model)

## Acknowledgments

- **Dataset Collection**: Footbonaut indoor training facility (multi-camera setup)
- **Model Baselines**: Ultralytics YOLOv8, YOLOv11, YOLOv26 architectures
- **Evaluation Framework**: COCO metrics, Normalized Wasserstein Distance
- **Inspiration**: SoccerNet, DFL Bundesliga, and ISSIA-CNR sports datasets

---

**Last Updated**: 2026-04-06  
**Version**: 1.0  
**Status**: Complete
```

I've created a **comprehensive, detailed README** for your ProxiBall Dataset Ablation Study. Here's what's included:

### Key Sections:

1. **Overview** - Clear problem statement about the domain gap
2. **Repository Structure** - Complete file organization
3. **Evaluation Scripts** - Detailed description of all 11 scripts organized by stages
4. **Requirements** - System and Python dependencies
5. **Dataset Format** - YOLO annotation format and bucket classification
6. **Quick Start** - Step-by-step setup and execution guide
7. **Metrics Explanation** - What mAP50, NWD-mAP, RMSE mean
8. **Results Summary** - Actual performance comparison tables with your data
9. **Advanced Features** - Memory optimization, stratification, visualization
10. **Ablation Methodology** - Scientific approach to ablation studies
11. **Citation & License** - Proper attribution format

The README is based on:
- Actual script files in your repo
- Requirements from `requirements.txt`
- Insights from `paper.md` about your methodology
- Test distribution data from your scripts

You can now use this to update your README by copying it to your repository!
