import pandas as pd
import numpy as np

# 读取CSV文件（第一行是列名）
df = pd.read_csv('subset/respective_ceemdan_se144hubei_pred.csv')

# 计算三列的和
df['sum'] = df['co-imf0'] + df['co-imf1'] + df['co-imf2']

# 创建输出DataFrame（序号和和值）
output_df = pd.DataFrame({
    'index': df.index,  # 序号从0开始
    'sum_value': df['sum']
})

# 保存到CSV文件（不包含表头）
output_df.to_csv('output144hubei.csv', index=False, header=False)

print("处理完成！")
print("\n输出结果：")
print(output_df)