import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all(title, pred_data, orig_data, dates):
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    plt.plot(dates, orig_data, 's-', label='Original Data', 
             linewidth=1.5, markersize=3, color='#1f77b4', alpha=0.8)
    plt.plot(dates, pred_data, 'o-', label='Forecasting Data', 
             linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Original vs Forecasting', fontsize=14, fontweight='bold')
    
    step = max(1, len(dates) // 10)
    xticks_idx = np.arange(0, len(dates), step)
    plt.xticks(xticks_idx, [dates[i] for i in xticks_idx], rotation=45, ha='right')
    
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower()}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
file_path = 'output145CEA.csv'
df_pred = pd.read_csv(file_path)
pred_data = df_pred.iloc[:, 1].values

demo_path = 'CEAdemo_data.csv'
if os.path.exists(demo_path):
    df_orig = pd.read_csv(demo_path, header=None)
    
    if df_orig.shape[1] >= 2:
        dates_all = df_orig.iloc[:, 0].values
        values_all = df_orig.iloc[:, 1].values
        
        if len(values_all) >= 100:
            orig_data = values_all[-100:]
            dates = dates_all[-100:]
        else:
            orig_data = values_all
            dates = dates_all
        
        min_len = min(len(pred_data), len(orig_data))
        pred_data = pred_data[:min_len]
        orig_data = orig_data[:min_len]
        dates = dates[:min_len]
        
        plot_all('method1', pred_data, orig_data, dates)