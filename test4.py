# 统计检验
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# 读取数据
def load_data():
    # 读取原始数据
    actual_data = pd.read_csv('guanzhou.csv', parse_dates=['date'])
    actual_data = actual_data.sort_values('date')  # 按日期排序
    actual_prices = actual_data['close'].values
    
    # 读取后100天的实际值
    actual_last_100 = actual_prices[-100:]
    
    # 读取两个模型的预测值（预测值在第二列）
    model1_data = pd.read_csv('output144guanzhou.csv', header=None)
    model2_data = pd.read_csv('compare/test_predictions.csv', header=None)
    
    model1_pred = model1_data.iloc[:, 1].values  # 第二列（索引1）是预测值
    model2_pred = model2_data.iloc[:, 1].values  # 第二列（索引1）是预测值
    
    return actual_last_100, model1_pred, model2_pred

# ADF检验（单位根检验）
def adf_test(series, series_name):
    """
    执行ADF检验
    H0: 存在单位根（非平稳）
    H1: 不存在单位根（平稳）
    """
    result = adfuller(series, autolag='AIC')
    
    print(f'\n{"="*50}')
    print(f'{series_name} - ADF检验结果')
    print(f'{"="*50}')
    print(f'ADF统计量: {result[0]:.6f}')
    print(f'p值: {result[1]:.6f}')
    print(f'临界值:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.6f}')
    
    if result[1] < 0.05:
        print(f'结论: 在95%置信水平下拒绝原假设，序列{series_name}是平稳的')
    else:
        print(f'结论: 在95%置信水平下不能拒绝原假设，序列{series_name}是非平稳的')
    
    return result

# Ljung-Box检验（白噪声检验）
def ljung_box_test(series, series_name, lags=[5, 10, 15, 20]):
    """
    执行Ljung-Box检验
    H0: 序列是白噪声（没有自相关）
    H1: 序列不是白噪声（存在自相关）
    """
    print(f'\n{"="*50}')
    print(f'{series_name} - Ljung-Box检验结果')
    print(f'{"="*50}')
    
    # 执行Ljung-Box检验
    lb_result = acorr_ljungbox(series, lags=lags, return_df=True)
    
    print(f'\n滞后阶数\tLB统计量\tp值\t\t结论')
    print(f'-' * 60)
    for i, lag in enumerate(lags):
        p_value = lb_result['lb_pvalue'].iloc[i]
        lb_stat = lb_result['lb_stat'].iloc[i]
        conclusion = '拒绝H0(非白噪声)' if p_value < 0.05 else '不能拒绝H0(白噪声)'
        print(f'{lag}\t\t{lb_stat:.6f}\t{p_value:.6f}\t{conclusion}')
    
    return lb_result

# DM检验（Diebold-Mariano检验）
def dm_test(actual, pred1, pred2, h=1):
    """
    执行Diebold-Mariano检验
    H0: 两个模型的预测精度相同
    H1: 两个模型的预测精度不同
    
    参数:
    actual: 实际值序列
    pred1: 模型1的预测值
    pred2: 模型2的预测值
    h: 预测步长（默认为1）
    """
    from scipy import stats
    
    # 计算预测误差
    e1 = actual - pred1
    e2 = actual - pred2
    
    # 计算损失函数（这里使用平方误差）
    loss1 = e1 ** 2
    loss2 = e2 ** 2
    
    # 损失差异
    d = loss1 - loss2
    
    # 计算DM统计量
    n = len(d)
    d_mean = np.mean(d)
    
    # 计算自协方差（考虑h步预测）
    gamma = []
    for k in range(0, h):
        if k == 0:
            gamma.append(np.mean((d - d_mean) ** 2))
        else:
            gamma.append(2 * np.mean((d[:-k] - d_mean) * (d[k:] - d_mean)))
    
    # 方差估计
    var_d = np.sum(gamma) / n
    
    # DM统计量
    dm_stat = d_mean / np.sqrt(var_d) if var_d > 0 else 0
    
    # 计算p值（双尾检验）
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    print(f'\n{"="*50}')
    print(f'Diebold-Mariano检验结果')
    print(f'{"="*50}')
    print(f'DM统计量: {dm_stat:.6f}')
    print(f'p值: {p_value:.6f}')
    
    if p_value < 0.05:
        print(f'结论: 在95%置信水平下拒绝原假设，两个模型的预测精度存在显著差异')
        if d_mean < 0:
            print(f'模型1的预测误差显著小于模型2，模型1更优')
        else:
            print(f'模型2的预测误差显著小于模型1，模型2更优')
    else:
        print(f'结论: 在95%置信水平下不能拒绝原假设，两个模型的预测精度没有显著差异')
    
    return dm_stat, p_value

# 计算各种误差指标
def calculate_metrics(actual, pred1, pred2):
    """
    计算常用的预测误差指标
    """
    e1 = actual - pred1
    e2 = actual - pred2
    
    metrics = {}
    
    # 均方误差 (MSE)
    metrics['MSE_model1'] = np.mean(e1 ** 2)
    metrics['MSE_model2'] = np.mean(e2 ** 2)
    
    # 均方根误差 (RMSE)
    metrics['RMSE_model1'] = np.sqrt(metrics['MSE_model1'])
    metrics['RMSE_model2'] = np.sqrt(metrics['MSE_model2'])
    
    # 平均绝对误差 (MAE)
    metrics['MAE_model1'] = np.mean(np.abs(e1))
    metrics['MAE_model2'] = np.mean(np.abs(e2))
    
    # 平均绝对百分比误差 (MAPE)
    epsilon = 1e-10
    metrics['MAPE_model1'] = np.mean(np.abs(e1 / (actual + epsilon))) * 100
    metrics['MAPE_model2'] = np.mean(np.abs(e2 / (actual + epsilon))) * 100
    
    return metrics

# 主函数
def main():
    print("开始加载数据...")
    
    # 加载数据
    actual, model1_pred, model2_pred = load_data()
    
    # 检查数据长度
    print(f"\n数据长度检查:")
    print(f"实际值长度: {len(actual)}")
    print(f"模型1预测值长度: {len(model1_pred)}")
    print(f"模型2预测值长度: {len(model2_pred)}")
    
    # 确保长度一致
    min_len = min(len(actual), len(model1_pred), len(model2_pred))
    if min_len < len(actual):
        print(f"警告：数据长度不一致，将截取到最短长度 {min_len}")
        actual = actual[:min_len]
        model1_pred = model1_pred[:min_len]
        model2_pred = model2_pred[:min_len]
    
    # 1. ADF检验（只检验实际收盘价）
    print("\n" + "="*60)
    print("1. ADF检验（单位根检验）")
    print("="*60)
    adf_test(actual, "实际收盘价")
    
    # 2. Ljung-Box检验（只检验实际收盘价）
    print("\n" + "="*60)
    print("2. Ljung-Box检验（白噪声检验）")
    print("="*60)
    ljung_box_test(actual, "实际收盘价")
    
    # 3. DM检验（比较两个模型的预测精度）
    print("\n" + "="*60)
    print("3. Diebold-Mariano检验（模型比较）")
    print("="*60)
    dm_test(actual, model1_pred, model2_pred, h=1)
    
    # 4. 计算误差指标
    print("\n" + "="*60)
    print("4. 预测误差指标对比")
    print("="*60)
    metrics = calculate_metrics(actual, model1_pred, model2_pred)
    
    print(f"\n{'指标':<20} {'模型1':<20} {'模型2':<20}")
    print("-" * 60)
    print(f"{'MSE':<20} {metrics['MSE_model1']:<20.6f} {metrics['MSE_model2']:<20.6f}")
    print(f"{'RMSE':<20} {metrics['RMSE_model1']:<20.6f} {metrics['RMSE_model2']:<20.6f}")
    print(f"{'MAE':<20} {metrics['MAE_model1']:<20.6f} {metrics['MAE_model2']:<20.6f}")
    print(f"{'MAPE(%)':<20} {metrics['MAPE_model1']:<20.6f} {metrics['MAPE_model2']:<20.6f}")
    
    # 5. 基本统计描述
    print("\n" + "="*60)
    print("5. 数据基本统计信息")
    print("="*60)
    print(f"\n实际值范围: [{actual.min():.2f}, {actual.max():.2f}]")
    print(f"实际值均值: {actual.mean():.2f}")
    print(f"实际值标准差: {actual.std():.2f}")
    
    # 计算预测误差用于统计
    residuals1 = actual - model1_pred
    residuals2 = actual - model2_pred
    
    print(f"\n模型1预测误差: 均值={np.mean(residuals1):.6f}, 标准差={np.std(residuals1):.6f}")
    print(f"模型1预测误差范围: [{residuals1.min():.6f}, {residuals1.max():.6f}]")
    
    print(f"\n模型2预测误差: 均值={np.mean(residuals2):.6f}, 标准差={np.std(residuals2):.6f}")
    print(f"模型2预测误差范围: [{residuals2.min():.6f}, {residuals2.max():.6f}]")

if __name__ == "__main__":
    main()