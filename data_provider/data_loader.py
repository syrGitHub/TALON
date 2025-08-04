import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
from data_provider.complexity import calculate_complexity
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', llm_model=None, size=None, data_path='ETTh1.csv',
                 scale=True, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.llm_model = llm_model
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}_{self.llm_model}.pt'))
        # print(self.data_stamp.shape, os.path.join(self.root_path, f'{data_name}.pt'))   # torch.Size([17420, 4096])
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print(self.data_stamp.shape, self.data_x.shape,
              self.data_y.shape)  # torch.Size([3552, 4096]) (3552, 7) (3552, 7)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # reshape seq_x to [token_num, token_len]
        patch = seq_x.squeeze(-1).reshape(self.token_num, self.token_len)  # [token_num, token_len]
        complexity = np.stack([calculate_complexity(p) for p in patch], axis=0)  # [token_num, 3]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, complexity

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', llm_model=None, size=None, data_path='ETTh1.csv',
                 scale=True, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.llm_model = llm_model
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}_{self.llm_model}.pt'))
        # print(self.data_stamp.shape, os.path.join(self.root_path, f'{data_name}.pt'))   # torch.Size([17420, 4096])
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print(self.data_stamp.shape, self.data_x.shape,
              self.data_y.shape)  # torch.Size([3552, 4096]) (3552, 7) (3552, 7)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # reshape seq_x to [token_num, token_len]
        patch = seq_x.squeeze(-1).reshape(self.token_num, self.token_len)  # [token_num, token_len]
        complexity = np.stack([calculate_complexity(p) for p in patch], axis=0)  # [token_num, 3]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, complexity

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', llm_model=None, size=None, data_path='ETTh1.csv',
                 scale=True, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.llm_model = llm_model
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}_{self.llm_model}.pt'))
        # print(self.data_stamp.shape, os.path.join(self.root_path, f'{data_name}.pt'))   # torch.Size([17420, 4096])
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print(self.data_stamp.shape, self.data_x.shape,
              self.data_y.shape)  # torch.Size([3552, 4096]) (3552, 7) (3552, 7)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # reshape seq_x to [token_num, token_len]
        patch = seq_x.squeeze(-1).reshape(self.token_num, self.token_len)  # [token_num, token_len]
        complexity = np.stack([calculate_complexity(p) for p in patch], axis=0)  # [token_num, 3]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, complexity

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Patch_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        data_name = self.data_path.split('.')[0]

        if data_name == 'ETTh1' or 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif data_name == 'ETTm1' or 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        # 获取整个window
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]

        # print(seq_x.shape)
        # reshape seq_x to [token_num, token_len]
        patch = seq_x.squeeze(-1).reshape(self.token_num, self.token_len)  # [token_num, token_len]
        # print(patch.shape)

        # 1. 全局窗口（这里默认全局窗口是当前patch所在seq）
        x_global_start = self.data_stamp[s_begin]
        x_global_end = self.data_stamp[s_end - 1] if (s_end - 1) < self.tot_len else self.data_stamp[-1]

        prompts = []
        for i, p in enumerate(patch):
            # 2. patch complexity
            complexity = calculate_complexity(p)
            trend, deriv, autocorr = [round(float(x), 3) for x in complexity]

            # 3. patch 窗口
            x_start_idx = s_begin + i * self.token_len
            x_end_idx = x_start_idx + self.token_len - 1
            x_start = self.data_stamp[x_start_idx]
            x_end = self.data_stamp[min(x_end_idx, len(self.data_stamp) - 1)]

            # 4. Prompt模板拼接
            prompt = f"""[Expert Routing Hint]
                        - The available expert types are: Linear, CNN, LSTM.

                        [Patch Time Context]
                        - This patch consists of {self.token_len} time steps, from {x_start} to {x_end}.
                        - It is part of a longer input window, which spans from {x_global_start} to {x_global_end} and contains {self.seq_len} time steps.

                        [Complexity Features]
                        - This patch has a trend strength of {trend:.3f}, local variation {deriv:.3f}, and autocorrelation {autocorr:.3f}.
                        """

            prompts.append(prompt)
        # print(len(prompts), len(prompts[0]))
        return prompts  # list, [token_num, str]

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

