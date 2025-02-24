# -*- coding: UTF-8 -*-

import os
import random
import logging
import torch
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, NoReturn, Any

device = torch.device('cuda')

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def df_to_dict(df: pd.DataFrame) -> dict:
    res = df.to_dict('list')
    for key in res:
        res[key] = np.array(res[key])
    return res


def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device)
    return batch


def check(check_list: List[tuple]) -> NoReturn:
    # observe selected tensors during training.
    logging.info('')
    for i, t in enumerate(check_list):
        d = np.array(t[1].detach().cpu())
        logging.info(os.linesep.join(
            [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
        ) + os.linesep)


def eval_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: eval(str(x)))  # some list-value columns
    return df


def format_metric(result_dict: Dict[str, Any]) -> str:
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys() if '@' in k])
    if not len(topks):
        topks = ['All']
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name] if topk != 'All' else result_dict[metric]
            if type(m) is float or type(m) is np.float64 or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int32 or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)


def non_increasing(lst: list) -> bool:
    return all(x >= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:])) # update the calculation of non_increasing to fit ealry stopping, 2023.5.14, Jiayu Li


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import inspect
import os

ROOT_PATH = 'C:/codes/ReChorus_visual'
def print_log(str):
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    file_name = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno  
    formatted_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{str}, {file_name}-Line{line_number}, Time{formatted_now}.')

def write_log(str, log_file_name='log.txt'):
    print_log(str)
    log_file_path = os.path.join(ROOT_PATH, log_file_name)
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    file_name = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno  
    formatted_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, 'a') as log_file:
        print(f'{str}, {file_name}-Line{line_number}, Time{formatted_now}.', file=log_file)

def write_test_result(str, test_result_name='test_result.txt'):
    test_result_path = os.path.join(ROOT_PATH, test_result_name)
    with open(test_result_path, 'a') as test_result:
        print(str, file=test_result)


import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def draw_points(X_tensor):
    # X = np.random.randn(1000, 64)

    X = X_tensor.detach().cpu().numpy()

    # 数据标准化（推荐预处理步骤）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用t-SNE进行降维
    tsne = TSNE(
        n_components=2,      # 降维到2维
        random_state=42,     # 随机种子保证可重复性
        perplexity=30,       # 建议值在5-50之间，根据数据量调整
        learning_rate=200,  # 学习率通常设置在10-1000之间
        n_iter=1000         # 迭代次数
    )
    X_2d = tsne.fit_transform(X_scaled)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                alpha=0.6,    # 设置透明度
                edgecolors='w', # 点边缘颜色
                s=40)         # 点大小

    plt.title('2D Visualization using t-SNE', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)      # 添加半透明网格
    plt.show()
    xxx = 0

# 输入：256*20*256的数
# 输出：每一个数，按256展开成一个频谱
def draw_frequency(period_torch, frequency_torch, valid_torch, u_id):

    frequency = frequency_torch.detach().cpu().numpy()
    valid = valid_torch.detach().cpu().numpy()
    idx = -1
    x = list(range(1, 257))
    count = 0
    for i, session, valid_tag in zip(range(256), frequency, valid):
        if valid_tag[-1] == 1 and u_id == 513:
            # for j in session:
            #     plt.plot(x, j, marker='o', linestyle='-', color='#FF6B6B')
            # plt.show()
            idx = i
            break

    return idx


def draw_weight_time(idx_session, numda_torch, current_interval_torch):

    current_interval = current_interval_torch.detach().cpu().numpy()
    numda = numda_torch.detach().cpu().numpy()

    x = current_interval[idx_session]
    y = numda[idx_session]
    result_2d = np.column_stack((x, y))
    plt.scatter(result_2d[:, 0], result_2d[:, 1], edgecolors='w', s=40)
    plt.show()
    return 0


def draw_weight_sim(idx_session, numda_torch, sim_data_torch):
    sim_data = sim_data_torch.detach().cpu().numpy()
    numda = numda_torch.detach().cpu().numpy()

    x = sim_data
    y = numda[idx_session]

    result_2d = np.column_stack((x, y))
    plt.scatter(result_2d[:, 0], result_2d[:, 1], edgecolors='w', s=40)
    plt.show()
    return 0


def draw_weight_sim_time(idx_session, current_interval_torch, sim_data_torch):
    current_interval = current_interval_torch.detach().cpu().numpy()
    sim_data = sim_data_torch.detach().cpu().numpy()


    x = current_interval
    y = sim_data

    result_2d = np.column_stack((x, y))
    plt.scatter(result_2d[:, 0], result_2d[:, 1], edgecolors='w', s=40)
    plt.show()
    return 0
