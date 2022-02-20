import os

import pandas as pd

from gen_table.my_table import MyTable


class ParamTable(MyTable):
    def __init__(self):
        super().__init__("params")

    def get_table_data(self):
        dic = {
            "NAC": 0.75,
            "NBC": 0,
            "SNAC": 0,
            "KMNC": 1000,
            "TKNC": 1,
            "LSC": "(1000, 100 / 2000)"
        }
        df = pd.DataFrame(dic, index=["RQ1 RQ2"])
        df.to_csv(os.path.join(self.output_tab_dir_path, "config_params.csv"))


class DopTable(MyTable):
    def __init__(self):
        super().__init__("dop")

    def get_table_data(self):
        abbr_arr = ['SF', 'ZM', 'BR', 'RT', 'NS', 'BL', 'SR']
        full_arr = ["shift", 'zoom', 'brightness', 'rotation', 'noise', 'blur', 'shear']
        name_arr = ["平移", "放缩", "亮度", "旋转", "噪声", "模糊", "错切"]

        csv_data = {
            "缩写": abbr_arr,
            "全称": full_arr,
            "名字": name_arr
        }
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(self.output_tab_dir_path, "config_dop.csv"), index=False)


class AdvTable(MyTable):
    def __init__(self):
        super().__init__("dop")

    def get_table_data(self):
        abbr_arr = ["bim", "pgd", "jsma", "ead", "fgsm"]
        link_arr = ["https://arxiv.org/abs/1607.02533", "https://arxiv.org/abs/1607.02533",
                    "https://arxiv.org/abs/1511.07528",
                    "https://arxiv.org/abs/1709.04114", "https://arxiv.org/abs/1607.02533"]

        csv_data = {
            "名称": abbr_arr,
            "连接": link_arr,
        }
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(self.output_tab_dir_path, "config_adv.csv"), index=False)


if __name__ == '__main__':
    ParamTable().get_table_data()
    DopTable().get_table_data()
    AdvTable().get_table_data()
