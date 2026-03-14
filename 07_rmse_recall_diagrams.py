import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_final_plots(recall_csv, rmse_csv, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black'
    })
    
    df_recall = pd.read_csv(recall_csv)
    df_rmse = pd.read_csv(rmse_csv)
    
    # Filter and order models
    target_models = ["ISSIA", "Yolo11s", "Test-Project", "Ball-Detection", "Soccernet", "ProxiBall"]
    
    # Filter dataframes
    df_recall = df_recall[df_recall['Model'].isin(target_models)]
    df_rmse = df_rmse[df_rmse['Model'].isin(target_models)]
    
    # Ensure exact order for Hue in bar charts and palette
    df_recall['Model'] = pd.Categorical(df_recall['Model'], categories=target_models, ordered=True)
    df_recall = df_recall.sort_values('Model')
    
    # Bar Charts (Melt format)
    melted_data = []
    for _, row in df_recall.iterrows():
        model = row['Model']
        for cat, buckets in [('Size', ['Small', 'Med', 'Large']), ('Velocity', ['Slow', 'Med', 'Fast'])]:
            for b_name in buckets:
                melted_data.append({
                    'Model': model, 
                    'Category': cat, 
                    'Bucket': b_name, 
                    'Recall': row[f'{cat}_{b_name}_Recall(%)']
                })
    df_long = pd.DataFrame(melted_data)
    df_long['Model'] = pd.Categorical(df_long['Model'], categories=target_models, ordered=True)
    
    # Color Palette
    # Consistent high-contrast color palette (matching core_metrics.py)
    # "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000", "#D55E00"
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000"]
    palette = {m: colors[i] for i, m in enumerate(target_models)}
    # Overwrite ProxiBall with Red if needed or keep consistent with colors list
    palette["ProxiBall"] = "#e74c3c" 
    
    # Bar Charts (Recall vs Bins)
    for cat_name, file_suffix in [('Velocity', '2a_Recall_Velocity'), ('Size', '2b_Recall_Size')]:
        plt.figure(figsize=(10, 6))
        order = ['Slow', 'Med', 'Fast'] if cat_name == 'Velocity' else ['Small', 'Med', 'Large']
        sns.barplot(data=df_long[df_long['Category'] == cat_name], 
                    x='Bucket', y='Recall', hue='Model', palette=palette, order=order)
        plt.title(f"Graph {file_suffix[:2]}: Recall vs. Ball {cat_name}", fontweight='bold', fontsize=16)
        plt.ylabel("Recall (%)", fontweight='bold')
        plt.xlabel(f"{cat_name} Category", fontweight='bold')
        plt.ylim(0, 105)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'Graph_{file_suffix}.png', dpi=300)
        plt.close()

    # Stratified Scatter Plots (RMSE vs Recall)
    categories = [
        ('Size', 'Small'), ('Size', 'Med'), ('Size', 'Large'),
        ('Velocity', 'Slow'), ('Velocity', 'Med'), ('Velocity', 'Fast')
    ]
    
    print("\nGenerating 6 stratified RMSE vs Recall graphs...")

    for idx, (cat, bucket) in enumerate(categories, start=4):
        plt.figure(figsize=(10, 8))
        
        # BOUNDARY LINES
        plt.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
        plt.axvline(x=100, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
        
        # Data for the current bucket
        recall_col = f"{cat}_{bucket}_Recall(%)"
        rmse_col = f"RMSE_{cat}_{bucket}"
        
        # Merge and enforce order
        df_plot = pd.merge(df_rmse[['Model', rmse_col]], df_recall[['Model', recall_col]], on='Model')
        df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=target_models, ordered=True)
        df_plot = df_plot.sort_values('Model')
        
        for i, row in df_plot.iterrows():
            model = row['Model']
            color = palette[model]
            size = 350 if model == 'ProxiBall' else 180
            zorder = 5 if model == 'ProxiBall' else 2
            
            x_val = row[recall_col]
            y_val = row[rmse_col]
            
            plt.scatter(x_val, y_val, color=color, s=size, edgecolors='black', alpha=1.0, zorder=zorder, label=model)
            
            # LABEL PLACEMENT
            if model == 'ProxiBall':
                ha, va = 'right', 'top'
                xy_off = (-5, -12)
                f_size = 16
            else:
                ha, va = 'center', 'bottom'
                xy_off = (0, 10)
                f_size = 14
                
            plt.annotate(model, (x_val, y_val), xytext=xy_off, textcoords='offset points', 
                         fontsize=f_size, ha=ha, va=va, fontweight='bold' if model == 'ProxiBall' else 'normal', zorder=6)
            
        plt.title(f"Graph 0{idx}: RMSE vs Recall ({bucket} {cat})", fontweight='bold', fontsize=18, pad=25)
        plt.xlabel(f"Recall on '{bucket}' {cat} (%) →", fontweight='bold', fontsize=13)
        plt.ylabel(f"RMSE (pixels) for '{bucket}' {cat} ←", fontweight='bold', fontsize=13)
        
        max_rmse = df_plot[rmse_col].max()
        ylim_top = max(5, max_rmse * 1.25)
        plt.ylim(0, ylim_top)
        plt.xlim(-5, 105)
        
        # IDEAL PERFORMANCE ZONE
        plt.text(98, ylim_top * 0.05, "IDEAL PERFORMANCE ZONE", color='darkgreen', fontsize=11, fontweight='bold', 
                 ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkgreen'))
        
        plt.grid(True, linestyle=':', alpha=0.4, zorder=1)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'Graph_0{idx}_RMSE_Recall_{cat}_{bucket}.png', dpi=300)
        plt.close()
        print(f"  [OK] Graph {idx}: {cat}_{bucket}")


if __name__ == "__main__":
    recall_file = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/03_stratified/03_stratified_iou_0.5.csv"
    rmse_file = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/06_rmse_and_cm/Table_2_RMSE_Stratified.csv"
    output_folder = "D:/Altay/dataset-evaluation/6.1.data-eval/outputs/07_final_graphs"
    
    generate_final_plots(recall_file, rmse_file, output_folder)