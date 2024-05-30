import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def fetch_stock_daily_k_data(
        stock_code, 
        stock_exchange='0',
        start_date='2018-01-01', 
        end_date=None, 
        fq=1, 
        other_info=None,
        folder_name='kline_data'):
    """
    获取指定股票代码的历史K线数据。
    :param stock_code: 股票代码，如'000001'（平安银行）
    :param stock_exchange: 股票交易所，0为沪市，1为深市，默认为0
    :param start_date: 开始日期，字符串格式，默认为'2018-01-01'
    :param end_date: 结束日期，字符串格式，默认为当前日期
    :param fq: 复权类型，1为前复权，2为后复权，默认为1
    :param other_info: 其他信息，行业、地域、概念，如提供则会保存到Excel文件中
    :param folder_name: 保存数据的文件夹名称，默认为'kline_data'
    :return: 包含历史K线数据的DataFrame
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    if end_date is None:  # 如果未指定结束日期，则默认为今天
        end_date = datetime.now().strftime('%Y%m%d')
    print(f"Fetching data from {start_date} to {end_date} for stock {stock_code}...")
    
    # 所需字段
    columns = [
        '股票代码', '股票名称', '日期', '开盘价', '收盘价', '最高价', '最低价',
        '成交量', '成交额', '振幅(%)', '涨跌幅(%)', '涨跌额', '换手率(%)',
    ]
    if other_info is not None:
        columns += ['行业', '地域', '概念', ]

    # 请求API获取部分数据
    base_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
    # secid字段：0.代表沪市，1.代表深市；.后面跟随股票代码
    secid_fields1 = f"secid={stock_exchange}.{stock_code}&fields1=f1,f2,f3,f4,f5,f6&"
    # fields2字段：f51: 日期；f52: 开盘价；f53: 收盘价；f54: 最高价；f55: 最低价；
    #   f56: 成交量；f57: 成交额；f58: 振幅；f59: 涨跌幅；f60: 涨跌额；f61: 换手率；
    fields2 = "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
    # klt字段：101代表日K线，102代表周K线，103代表月K线；fqt字段：1代表前复权，2代表后复权；beg和end字段：开始和结束日期
    klt_fqt_beg_end = f"klt=101&fqt={fq}&beg={start_date}&end={end_date}"
    url_api = base_url + secid_fields1 + fields2 + klt_fqt_beg_end
    response = requests.get(url_api)
    response.raise_for_status()
    data = response.json()
    # pprint(data)
    
    # 从API返回的数据中提取部分所需字段
    try:
        klines_data = data['data']['klines']
    except Exception as e:
        print(f"Error: {e}\nPlease check stock exchange or any other parameters.")
        return None
    stock_name = data['data']['name'].replace('*', '')  # 要去掉股票名中的星号
    klines_table = []
    for kline in klines_data:
        kline = kline.split(',')
        kline = [stock_code, stock_name] + kline
        if other_info is not None:
            kline += other_info
        klines_table.append(kline)
    # print(klines_table[:5])
    klines_table = pd.DataFrame(klines_table, columns=columns)

    # 保存数据到excel文件
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    try:
        klines_table.to_excel(f"{folder_name}/{stock_code}_{stock_name}_daily_k_data.xlsx", index=False)
        print(f"Data saved to {folder_name}/{stock_code}_{stock_name}_daily_k_data.xlsx successfully.\n")  
    except Exception as e:
        print(f"Error: {e}\nPlease check if the file is open and close it before running the script again.")
    return klines_table


fetch_stock_daily_k_data('300496', stock_exchange='0')


def fetch_category_stocks(category, crawl_all=False):
    """
    获取指定行业/概念/地域板块内所有股票的代码。
    :param category: 行业/概念/地域板块代码，如'0447'（互联网服务）、'0918'（特高压）
    :param crawl_all: 是否爬取所有股票的数据，默认为False
    :return: 行业/概念/地域板块内所有股票的代码列表
    """
    # pn: 页码；pz: 每页数量；po: 排序方式(0: 正序，1: 倒序)；fid: 排序字段
    base_url = "http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=0&np=1&fltt=2&invt=2&fid=f12&"
    # fs: 股票筛选条件；bk: 板块代码
    fs = f"fs=b:BK{category}&"
    # fields: 返回字段；f12: 股票代码；f13: 交易所，0为深证，1为上证；f14: 股票名称；f100: 行业；f102: 地域；f103: 概念
    fields = "fields=f12,f13,f14,f100,f102,f103"
    url = base_url + fs + fields
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # pprint(data)
    
    stocks = data['data']['diff']
    stocks_all_info = [list(stock.values()) for stock in stocks]
    # 股票代码开头900、200为B股，5位为H股，需要排除
    stocks_all_info = [stock for stock in stocks_all_info if not stock[0].startswith(('900', '200')) and len(stock[0]) == 6]
    # print(stocks_all_info[0])
    stock_codes = [stock[0] for stock in stocks_all_info]  # 股票代码
    if not crawl_all:
        return stock_codes
    
    # 收集股票信息，爬取所有股票的数据
    stock_exchanges = [stock[1] for stock in stocks_all_info]  # 交易所
    stock_info = [[stock[3], stock[4], stock[5]] for stock in stocks_all_info]  # 行业、地域、概念
    len_stock_codes = len(stock_codes)
    for i in range(len_stock_codes):
        print(f"[{i+1}/{len_stock_codes}]", end=" ")
        fetch_stock_daily_k_data(
            stock_codes[i], stock_exchange=stock_exchanges[i], other_info=stock_info[i],
            folder_name=f"BK{category}_kline_data"
        )
    return stock_codes


Industry = "0447"  # 互联网服务
fetch_category_stocks(Industry, crawl_all=True)


def convert_daily_to_yearly(daily_kline_folder):
    """
    将日K线数据转换为年数据。
    :param daily_kline_folder: 存放日K线数据的文件夹路径
    :return: 包含年数据的DataFrame
    """
    # 返回数据的字段
    columns = [
        '股票代码', '股票名称', '年份', '开盘价', '收盘价', '最高价', '最低价',
        '成交量', '成交额', '振幅(%)', '涨跌幅(%)', '涨跌额', '换手率(%)',
        '行业', '地域', '概念', 
    ]
    # 为返回数据创建空DataFrame
    yearly_kline = pd.DataFrame(columns=columns)

    # 统计日K线数据文件夹中的所有表格
    file_list = []
    for file in os.listdir(daily_kline_folder):
        if file.endswith('daily_k_data.xlsx'):
            file_list.append(file)
    file_num = len(file_list)
    print(f"Found {file_num} daily kline data files.")

    # 对每个表格内的数据进行处理
    for num, file in enumerate(file_list):
        # 读取表格数据
        file_path = os.path.join(daily_kline_folder, file)
        print(f"\n[{num+1}/{file_num}] Converting {file_path} to yearly data...", end=" ")
        df = pd.read_excel(file_path, dtype={'股票代码': str})

        # 提取年份，有些股票的数据可能不是从2018年开始的
        df['年份'] = df['日期'].str[:4]
        years = df['年份'].unique()
        print(f"Years: {years}")

        for year in years:
            # 提取日K线数据中，指定年份里的所有数据
            df_year = df[df['年份'] == year]
            # 开盘价、收盘价：年初和年末的第一天和最后一天的数据
            open_price = df_year['开盘价'].values[0]
            close_price = df_year['收盘价'].values[-1]
            # 最高价、最低价：全年的最高和最低价
            max_price = df_year['最高价'].max()
            min_price = df_year['最低价'].min()
            # 成交量、成交额：全年的总和
            volume = df_year['成交量'].sum()
            amount = df_year['成交额'].sum()
            # 其他需要计算的数据
            amplitude = (max_price - min_price) / min_price * 100  # 振幅
            price_change = close_price - open_price  # 涨跌额
            price_limit = price_change / open_price * 100  # 涨跌幅
            yearly_data = [
                df_year['股票代码'].values[0],
                df_year['股票名称'].values[0],
                year,
                open_price, close_price, max_price, min_price,
                volume, amount, amplitude,
                price_limit, price_change,
                df_year['换手率(%)'].mean(),  # 换手率取平均值
                df_year['行业'].values[0],
                df_year['地域'].values[0],
                df_year['概念'].values[0],
            ]
            yearly_kline.loc[len(yearly_kline)] = yearly_data

    # 将结果保存到Excel文件
    yearly_kline.to_excel(f"{daily_kline_folder}/yearly_kline_data.xlsx", index=False)


convert_daily_to_yearly('kline_data')


def fetch_category_weekly_k_data(
        category_code, 
        start_date='2018-01-01', 
        end_date=None, 
        fq=1):
    """
    获取指定行业/概念/地域板块的周K线数据。
    :param category_code: 行业/概念/地域板块代码，如'0447'（互联网服务）、'0918'（特高压）
    :param start_date: 开始日期，字符串格式，默认为'2018-01-01'
    :param end_date: 结束日期，字符串格式，默认为当前日期
    :param fq: 复权类型，1为前复权，2为后复权，默认为1
    :return: 包含周K线数据的DataFrame
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    if end_date is None:  # 如果未指定结束日期，则默认为今天
        end_date = datetime.now().strftime('%Y%m%d')
    print(f"Fetching data from {start_date} to {end_date} for category {category_code}...")
    
    # 所需字段
    columns = [
        '板块代码', '板块名称', '日期', '开盘价', '收盘价', '最高价', '最低价',
        '成交量', '成交额', '振幅(%)', '涨跌幅(%)', '涨跌额', '换手率(%)',
    ]

    # 请求API获取部分数据
    base_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
    # secid字段：90.代表板块，.后面跟随板块代码
    secid_fields1 = f"secid=90.BK{category_code}&fields1=f1,f2,f3,f4,f5,f6&"
    # fields2字段：f51: 日期；f52: 开盘价；f53: 收盘价；f54: 最高价；f55: 最低价；
    #   f56: 成交量；f57: 成交额；f58: 振幅；f59: 涨跌幅；f60: 涨跌额；f61: 换手率；
    fields2 = "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
    klt_fqt_beg_end = f"klt=102&fqt={fq}&beg={start_date}&end={end_date}"  # 周K，klt=102
    url_api = base_url + secid_fields1 + fields2 + klt_fqt_beg_end
    print(f"Fetching from URL: {url_api}")
    response = requests.get(url_api)
    response.raise_for_status()
    data = response.json()
    
    # 从API返回的数据中提取部分所需字段
    try:
        klines_data = data['data']['klines']
    except Exception as e:
        print(f"Error: {e}\nPlease check category code or any other parameters.")
        return None
    # print(klines_data)
    category_code = data['data']['code']
    category_name = data['data']['name']
    klines_table = []
    for kline in klines_data:
        kline = kline.split(',')
        kline = [category_code, category_name] + kline
        klines_table.append(kline)
    # print(klines_table[:5])
    klines_table = pd.DataFrame(klines_table, columns=columns)

    # 保存数据到excel文件
    try:
        klines_table.to_excel(f"{category_code}_{category_name}_weekly_k_data.xlsx", index=False)
        print(f"Data saved to {category_code}_{category_name}_weekly_k_data.xlsx successfully.\n")  
    except Exception as e:
        print(f"Error: {e}\nPlease check if the file is open and close it before running the script again.")
    return klines_table


fetch_category_weekly_k_data('1162')


def load_and_preprocess_data(yearly_kline_file):
    """读取数据并进行预处理"""
    df = pd.read_excel(yearly_kline_file)
    features = df[['涨跌幅(%)', '振幅(%)']]
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    return data_scaled, df


def plot_elbow_curve(data_scaled):
    """绘制肘部法则曲线"""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()


yearly_kline = 'kline_data/yearly_kline_data.xlsx'
data_scaled, data_frame = load_and_preprocess_data(yearly_kline)
plot_elbow_curve(data_scaled)


def create_and_visualize_kmeans_model(data_scaled, df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_scaled)
    
    # 聚类可视化
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        cluster = df[df['Cluster'] == i]
        plt.scatter(cluster['振幅(%)'], cluster['涨跌幅(%)'], label=f'Cluster {i}')
    plt.colorbar()
    
    plt.title('Clusters of Stocks')
    plt.xlabel('Amplitude (%)')
    plt.ylabel('Price Change (%)')
    plt.legend()
    plt.show()

    # 分析每个簇的特性
    print(df.groupby('Cluster')[['涨跌幅(%)', '振幅(%)']].describe().T)


n_clusters = 5
create_and_visualize_kmeans_model(data_scaled, data_frame, n_clusters)
