import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_dataset_geometry(labels_dir, out_dir, img_w=960, img_h=960):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    label_paths = list(Path(labels_dir).glob('*.txt'))
    
    # Списки для сбора статистики
    x_centers = []
    y_centers = []
    areas_px = []
    aspect_ratios = []
    
    print(f"Анализ {len(label_paths)} файлов разметки...")

    for txt_path in label_paths:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # Читаем нормализованные YOLO-координаты
                _, x_c, y_c, w, h = map(float, parts[:5])
                
                # Переводим в абсолютные пиксели
                abs_x = x_c * img_w
                abs_y = y_c * img_h
                abs_w = w * img_w
                abs_h = h * img_h
                
                # 1. Позиции для хитмапа
                x_centers.append(abs_x)
                y_centers.append(abs_y)
                
                # 2. Площадь бокса в пикселях (для Size Distribution)
                areas_px.append(abs_w * abs_h)
                
                # 3. Соотношение сторон (Aspect Ratio = W / H)
                if h > 0:
                    aspect_ratios.append(w / h)

    if not x_centers:
        print("ОШИБКА: Лейблы не найдены или файлы пусты.")
        return

    # Настройки стиля для научных статей
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    print(f"Собрано {len(x_centers)} боксов. Генерируем графики...")

    # ==========================================
    # График 3a: Spatial Heatmap (Тепловая карта позиций)
    # ==========================================
    plt.figure(figsize=(10, 6))
    # Используем KDE (Kernel Density Estimate) для красивой тепловой карты
    sns.kdeplot(x=x_centers, y=y_centers, cmap="magma", fill=True, thresh=0.05, bw_adjust=0.5)
    
    plt.title('Dataset EDA: Spatial Heatmap of Ball Positions', fontweight='bold', pad=15)
    plt.xlabel('X Coordinate (Pixels)', fontweight='bold')
    plt.ylabel('Y Coordinate (Pixels)', fontweight='bold')
    
    # Устанавливаем лимиты по размеру кадра и ИНВЕРТИРУЕМ ось Y 
    # (в изображениях Y=0 находится сверху)
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0) 
    
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'EDA_1_Spatial_Heatmap.png', dpi=300)
    plt.close()

    # ==========================================
    # График 3b: Bounding Box Area Distribution
    # ==========================================
    plt.figure(figsize=(9, 6))
    sns.histplot(areas_px, bins=50, color='#3498db', kde=True, edgecolor='black')
    
    plt.title('Dataset EDA: Bounding Box Area Distribution', fontweight='bold', pad=15)
    plt.xlabel('Bounding Box Area ($Pixels^2$)', fontweight='bold')
    plt.ylabel('Frequency (Number of Frames)', fontweight='bold')
    
    # Добавим линию медианы, чтобы показать, что большинство мячей крошечные
    median_area = np.median(areas_px)
    plt.axvline(median_area, color='#e74c3c', linestyle='dashed', linewidth=2, 
                label=f'Median Area: {median_area:.0f} px²')
    
    # Limit X-axis to focus on micro-objects (up to 99th percentile or fixed 10000 px^2)
    plt.xlim(0, min(np.percentile(areas_px, 99) * 1.2, 10000))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'EDA_2_Area_Distribution.png', dpi=300)
    plt.close()

    # ==========================================
    # График 3c: Aspect Ratio Distribution
    # ==========================================
    plt.figure(figsize=(9, 6))
    sns.histplot(aspect_ratios, bins=40, color='#e67e22', kde=True, edgecolor='black')
    
    plt.title('Dataset EDA: Aspect Ratio Distribution (W/H)', fontweight='bold', pad=15)
    plt.xlabel('Aspect Ratio (Width / Height)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    
    # Отметим границу, где начинается "Fast" velocity (Aspect Ratio > 1.3)
    plt.axvline(1.3, color='#c0392b', linestyle='dashed', linewidth=2, 
                label='Motion Blur Threshold (> 1.3)')
    
    # Limit X-axis to focus on the bulk of the distribution (1.0 to 3.0)
    plt.xlim(0.8, 3.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'EDA_3_AspectRatio_Distribution.png', dpi=300)
    plt.close()

    print(f"\n[УСПЕХ] EDA графики сохранены в папку: {out_dir}")
    print(f"Медианная площадь бокса: {median_area:.1f} пикселей в квадрате.")
    print(f"Среднее соотношение сторон: {np.mean(aspect_ratios):.2f}")

if __name__ == "__main__":
    
    # Путь к Ground Truth лейблам вашего тестового сета (или можете натравить на весь train сет)
    labels = "D:/Altay/dataset-evaluation/datasets/dataset-main/train/labels"
    outputs = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/10_dataset_eda"
    
    analyze_dataset_geometry(labels, outputs)