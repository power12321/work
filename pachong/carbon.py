import requests
import pandas as pd
import time
from tqdm import tqdm

# 配置
BASE_URL = "https://www.hbets.cn/list_51.html?page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

def fetch_page_data_fast(page_num):
    """快速获取单页数据"""
    url = BASE_URL.format(page_num)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = 'utf-8'
        
        # 直接解析HTML表格，速度快
        dfs = pd.read_html(response.text)
        
        for df in dfs:
            if '日期' in str(df.columns):
                return df
        return None
    except Exception as e:
        return None

def main():
    print("=" * 50)
    print("湖北碳排放权交易中心 - 快速爬取")
    print("=" * 50)
    
    all_data = []
    page = 1
    
    # 先试探总页数（快速模式）
    print("正在探测总页数...")
    max_page = 500  # 设置一个上限
    for test_page in range(1, 20):
        url = BASE_URL.format(test_page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if '暂无数据' in resp.text or len(resp.text) < 5000:
                max_page = test_page - 1
                break
        except:
            pass
    
    print(f"预计共 {max_page} 页")
    
    # 批量爬取，减少延迟
    print("开始批量爬取...")
    with tqdm(total=max_page, desc="爬取进度") as pbar:
        for page in range(1, max_page + 1):
            df = fetch_page_data_fast(page)
            if df is not None and not df.empty:
                all_data.append(df)
            
            pbar.update(1)
            # 每10页才休息0.1秒，而不是每页都休息
            if page % 10 == 0:
                time.sleep(0.1)
    
    if not all_data:
        print("未获取到数据，尝试Selenium方案...")
        return
    
    # 合并数据
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 清洗数据
    final_df = final_df.dropna(how='all')
    if '日期' in final_df.columns:
        final_df['日期'] = pd.to_datetime(final_df['日期'], errors='coerce')
        final_df = final_df.dropna(subset=['日期'])
        final_df = final_df[(final_df['日期'] >= '2013-01-01') & (final_df['日期'] <= '2026-12-31')]
        final_df = final_df.sort_values('日期')
    
    # 保存
    output_file = 'hbets_all_data.csv'
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n完成！共 {len(final_df)} 条数据，保存至 {output_file}")

if __name__ == "__main__":
    main()