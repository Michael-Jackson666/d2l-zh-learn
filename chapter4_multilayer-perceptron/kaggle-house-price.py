import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# Part 1: 数据下载和预处理
# =============================================================================

# --- 数据下载辅助函数 ---
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

# --- 数据集URL注册 ---
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def preprocess_data():
    """下载并预处理房价数据集"""
    print("开始下载和预处理数据...")
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    
    # 删除ID列，合并特征
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # --- 数值特征处理 ---
    # 找出所有数值特征的列名
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # 标准化：减去均值，除以标准差
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 标准化后，均值为0，可以直接用0填充缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # --- 类别特征处理 (独热编码) ---
    all_features = pd.get_dummies(all_features, dummy_na=True)
    
    # --- 转换为张量 ---
    n_train = train_data.shape[0]
    # (修复!) 将布尔类型的独热编码特征转换为浮点数
    train_features = torch.tensor(all_features[:n_train].values.astype(np.float32), dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values.astype(np.float32), dtype=torch.float32)
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    
    print("数据预处理完成。")
    return train_features, test_features, train_labels, test_data


# =============================================================================
# Part 2: 模型、损失函数和训练逻辑
# =============================================================================

def get_net(in_features):
    """定义一个简单的线性模型作为基线"""
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels, loss):
    """计算对数均方根误差 (Log RMSE)"""
    # 为了在取对数时保证数值稳定，将小于1的预测值裁剪为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """通用的训练函数"""
    train_ls, test_ls = [], []
    # 兼容不同d2l版本，去掉num_workers参数
    try:
        train_iter = d2l.load_array((train_features, train_labels), batch_size, num_workers=0)
    except TypeError:
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # 计算并记录每个epoch的log_rmse
        train_ls.append(log_rmse(net, train_features, train_labels, loss))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels, loss))
            
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    """获取K折交叉验证的第i折数据"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    """执行K折交叉验证"""
    print(f"\n--- 开始 {k}-折交叉验证 ---")
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        in_features = X_train.shape[1]
        net = get_net(in_features)
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.show()
        print(f'折 {i + 1}，训练 log rmse: {train_ls[-1]:.4f}, 验证 log rmse: {valid_ls[-1]:.4f}')
    return train_l_sum / k, valid_l_sum / k

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    """在整个数据集上训练并生成提交文件"""
    print("\n--- 在完整训练集上训练并进行预测 ---")
    in_features = train_features.shape[1]
    net = get_net(in_features)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'最终训练 log rmse: {train_ls[-1]:.4f}')
    
    # --- 预测并保存 ---
    net.eval()
    preds = net(test_features).detach().numpy()
    # 保证Id列存在
    if 'Id' not in test_data.columns:
        # 若test_data没有Id列，则自动生成Id
        test_data['Id'] = np.arange(1461, 1461 + len(test_data))
    test_data['SalePrice'] = pd.Series(preds.reshape(-1))
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("\n已生成 submission.csv 文件！")

# =============================================================================
# Part 3: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 1. 获取数据 ---
    train_features, test_features, train_labels, test_data_with_id = preprocess_data()
    
    # --- 2. K-折交叉验证选择超参数 ---
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print(f'\n{k}-折验证平均结果:')
    print(f'  平均训练 log rmse: {train_l:.4f}')
    print(f'  平均验证 log rmse: {valid_l:.4f}')
    
    # --- 3. 在完整数据上训练并生成提交文件 ---
    # 使用通过交叉验证找到的（或你认为更好的）超参数
    final_num_epochs, final_lr, final_wd, final_bs = 100, 5, 0, 64
    train_and_pred(train_features, test_features, train_labels, test_data_with_id,
                   final_num_epochs, final_lr, final_wd, final_bs)