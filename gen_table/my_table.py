import abc

import pandas as pd
import os

from utils import model_conf


class MyTable(object):

    def __init__(self, prefix):
        self.base_path = "result"
        self.prefix = prefix
        self.mode_arr = ["dau", "adv"]
        self.output_tab_dir_path = "tabs"
        self.init_dirs()

    def init_dirs(self):
        os.makedirs(self.output_tab_dir_path, exist_ok=True)

    def get_dir_name(self, mode, data_name, model_name):
        return "{}_{}_{}_{}".format(self.prefix, mode, data_name, model_name)

    def get_dir_path(self, mode, data_name, model_name):
        dir_name = "{}_{}_{}_{}".format(self.prefix, mode, data_name, model_name)
        return os.path.join(self.base_path, dir_name)

    @staticmethod
    def add_df(df, csv_data):
        if df is None:  # 如果是空的
            df = pd.DataFrame(csv_data, index=[0])
        else:
            df.loc[df.shape[0]] = csv_data
        return df

    @staticmethod
    def num_to_str(num, trunc=2):
        return format(num, '.{}f'.format(trunc))

    @abc.abstractmethod
    def get_table_data(self):
        ...

    @staticmethod
    def get_df2tab_index_dict():
        params = {
            "cov_nac": "NAC",
            "cov_nbc": "NBC",
            "cov_snac": "SNAC",
            "cov_tknc": "TKNC",
            "cov_kmnc": "KMNC",
            "cov_lsc": "LSC",
            "sp_c_4": "Pt",
        }
        return params

    def get_table_frame(self):
        index_arr, columns = self.get_table_index_name_arr(), self.get_table_columns_arr()
        df = pd.DataFrame(
            index=index_arr,
            columns=[columns])
        return df

    def get_tab_index_by_df_columns(self, name):
        return self.get_df2tab_index_dict()[name]

    def get_df_columns_arr(self):
        return self.get_df2tab_index_dict().keys()

    def get_table_index_name_arr(self):
        return self.get_df2tab_index_dict().values()

    def get_table_columns_arr(self):
        res = []
        model_data = model_conf.model_data
        for (k, v_arr) in model_data.items():
            for v in v_arr:
                res.append(self.get_table_column(k, v))
        return res

    def get_table_column(self, data_name, model_name):
        return model_conf.get_pair_name(data_name, model_name)
