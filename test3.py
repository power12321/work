# 预测值与真实值对比图

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# 读取数据
#pred = pd.read_csv('output145CEA.csv', header=None)[1].values  # 预测值100个
pred = pd.read_csv('compare/test_predictions.csv', header=None)[1].values  # 预测值100个

true = pd.read_csv('CEAdemo_data.csv', header=None)[1].values[-100:]  # 真实值最后100个

# 计算指标
r2 = r2_score(true, pred)
rmse = mean_squared_error(true, pred, squared=False)
mae = mean_absolute_error(true, pred)
mape = mean_absolute_percentage_error(true, pred) * 100

print(f"R²: {r2:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MAPE: {mape:.2f}%")

# 画图
plt.figure(figsize=(14, 6))
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# 生成日期序列
dates = np.arange(len(true))

plt.plot(dates, true, 's-', label='Original Data', 
         linewidth=1.5, markersize=3, color='#1f77b4', alpha=0.8)
plt.plot(dates, pred, 'o-', label='Forecasting Data', 
         linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8)

plt.xlabel('Sample', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Original vs Forecasting', fontsize=14, fontweight='bold')

step = max(1, len(dates) // 10)
xticks_idx = np.arange(0, len(dates), step)
plt.xticks(xticks_idx, [dates[i] for i in xticks_idx])

plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("图片已保存为 comparison.png")