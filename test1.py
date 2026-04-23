# 主方法

import CEEMDAN_LSTM as cl

def run_respective_example():
    print('##################################')
    
    # (1) 声明路径
    print("\n(1) Declare a path for saving files:")
    print("-------------------------------")
    series = cl.declare_path()
    
    # (2) CEEMDAN 分解
    print("\n(2) CEEMDAN decompose:")
    print("-------------------------------")
    cl.declare_vars(mode='ceemdan')
    imfs = cl.emd_decom()
    
    # (3) 样本熵
    print("\n(3) Sample Entropy:")
    print("-------------------------------")
    cl.sample_entropy()
    
    # (4) 整合 IMFs
    print("\n(4) Integrating IMFs:")
    print("-------------------------------")
    cl.integrate(inte_form=[[0],[1,2,3,4],[5,6,7,8]])  # form='145'
    
    # (5) 分别 LSTM 预测
    print("\n(5) Forecast with Respective LSTM:")
    print("-------------------------------")
    cl.declare_vars(mode='ceemdan_se', form='144')  # 声明预测变量
    cl.Respective_LSTM()  # 使用 LSTM 预测

if __name__ == '__main__':
    run_respective_example()