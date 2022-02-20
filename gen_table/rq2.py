import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import model_conf
import pandas as pd

from gen_data.DauUtils import get_data_size
from gen_table.my_fig import MyFig
from gen_table.my_table import MyTable
import numpy as np


class RQ2Table(MyTable):
    def __init__(self):
        self.ps_file_name = "res_{}_max.csv"
        self.base_file_name = "res_{}_empty.csv"
        self.prefix = "selection"
        self.csv_file = "res_{}_all.csv"
        super().__init__(self.prefix)

    def get_output_csv_path(self, mode):
        output_csv_path = os.path.join(self.output_tab_dir_path, "{}_{}.csv".format(self.prefix, mode))
        print("output_csv_path", output_csv_path)
        return output_csv_path

    def get_table_frame2(self, mode):
        index_arr, columns = self.get_table_index_name_arr(), self.get_table_columns_arr()
        index_arr = list(index_arr)
        index_arr.remove("KMNC")
        columns1_arr = []
        columns2_arr = []
        if mode == "dau":
            columns2_base_arr = ["cam_max", "p_cam_max", "rate", "time", "p_acc", ]
        else:
            columns2_base_arr = ["cam_max", "p_cam_max", "rate", "time", "p_acc", "p_base_acc"]
        for i in columns:
            columns1_arr += [i] * len(columns2_base_arr)
            columns2_arr += columns2_base_arr
        df = pd.DataFrame(
            index=index_arr,
            columns=[columns1_arr, columns2_arr])
        return df

    def get_tab_index_by_df_columns(self, name):
        params = {
            "NAC": "NAC",
            "NBC": "NBC",
            "SNAC": "SNAC",
            "TKNC": "TKNC",
            "LSC": "LSC",
            "DeepSpace": "Pt",
        }
        return params[name]

    def get_table_data(self):
        for mode in self.mode_arr:
            df_res = self.get_table_frame2(mode)
            output_csv_path = self.get_output_csv_path(mode)
            for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
                for model_name in model_name_arr:
                    _, test_size = get_data_size(data_name)
                    if mode == "dau":
                        total_size = test_size * 4
                    else:
                        total_size = test_size * 2
                    dir_path = self.get_dir_path(mode, data_name, model_name)
                    if not os.path.exists(dir_path):
                        print(dir_path, "not exist")
                        continue
                    csv_path = os.path.join(dir_path, self.ps_file_name.format(mode))
                    empty_path = os.path.join(dir_path, self.base_file_name.format(mode))
                    if os.path.exists(csv_path):
                        df_data = pd.read_csv(csv_path)
                        if mode == "adv":
                            df_empty = pd.read_csv(empty_path)
                            p_base_acc = df_empty["p_acc_max"].values[0]
                        for index, row in df_data.iterrows():
                            lb = row["name"]
                            table_index = self.get_tab_index_by_df_columns(lb)
                            cam_max = row["cam_max"]
                            p_cam_max = cam_max / total_size
                            cam_rate = row["rate"]
                            time = row["time"]
                            acc = row["p_acc_max"]
                            table_column = self.get_table_column(data_name, model_name)
                            df_res.loc[table_index, (table_column, "cam_max")] = cam_max
                            df_res.loc[table_index, (table_column, "p_cam_max")] = float(
                                self.num_to_str(p_cam_max, 4)) * 100
                            df_res.loc[table_index, (table_column, "rate")] = self.num_to_str(cam_rate, 2)
                            df_res.loc[table_index, (table_column, "time")] = int(time)
                            df_res.loc[table_index, (table_column, "p_acc")] = int(float(self.num_to_str(acc, 2)) * 100)
                            if mode == "adv":
                                df_res.loc[table_index, (table_column, "p_base_acc")] = int(
                                    float(self.num_to_str(p_base_acc, 2)) * 100)
            df_res.to_csv(output_csv_path)


# max_size max_per per_retrain
class RQ2Fig(MyFig):
    def __init__(self, ):
        self.tab = RQ2Table()
        self.select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
        super().__init__(self.tab)

    def init_dirs(self):
        for mode in self.tab.mode_arr:
            output_base_path = self.get_output_fig_base_path(mode)
            os.makedirs(output_base_path, exist_ok=True)
        pass

    def get_output_fig_base_path(self, mode):
        output_base_path = os.path.join(self.output_fig_dir_path, self.tab.prefix, mode)
        return output_base_path

    def plot_figs(self):
        # k_arr = self.tab.get_df_columns_arr()
        for mode in tqdm(self.tab.mode_arr):
            for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
                for model_name in model_name_arr:
                    dir_path = self.tab.get_dir_path(mode, data_name, model_name)
                    if not os.path.exists(dir_path):
                        print(dir_path, "not exist")
                        continue
                    csv_path = os.path.join(dir_path, self.tab.csv_file.format(mode))
                    output_dir = self.get_output_fig_base_path(mode)
                    fig_name = self.get_figs_name(data_name, model_name)
                    self.plot_line_figs(csv_path, output_dir, fig_name)

    def plot_line_figs(self, csv_path, output_dir, fig_name):
        select_size_ratio_arr = np.array(self.select_size_ratio_arr) * 100
        name_arr = []
        df = pd.read_csv(csv_path)
        for k in df.columns:
            if "p_acc" in k:
                name_arr.append(k)
        for index, row in df.iterrows():
            # print(row)
            name = row["name"]
            # cl = color_dict[lb]
            res_arr = [row[x] * 100 for x in name_arr]
            lb = str(name).split("_")[0]
            lb = self.get_lb_dict()[lb]
            cl = self.get_color_dict()[lb]
            if "Pt" in name or "DeepSpace" in name:
                plt.plot(select_size_ratio_arr, res_arr, label=lb, color=cl, marker="o")
            elif 'Random' in name:
                plt.plot(select_size_ratio_arr, res_arr, label=lb, color=cl, marker="x")
            else:
                plt.plot(select_size_ratio_arr, res_arr, label=lb, color=cl, marker="x", alpha=0.8)
            # plt.plot(select_arr, res_arr, label=lb, color="crimson", marker="o")
        data_name, model_name = fig_name.split("_")
        if model_name == model_conf.LeNet5:
            model_name = "LeNet-5"
        elif model_name == model_conf.LeNet1:
            model_name = "LeNet-1"
        elif model_name == model_conf.resNet20:
            model_name = "ResNet-20"
        elif model_name == model_conf.vgg16:
            model_name = "VGG-16"
        else:
            raise ValueError()
        if data_name == model_conf.mnist:
            data_name = "MNIST"
        elif data_name == model_conf.fashion:
            data_name = "Fashion"
        elif data_name == model_conf.cifar10:
            data_name = "CIFAR"
        elif data_name == model_conf.svhn:
            data_name = "SVHN"
        else:
            raise ValueError()
        plt.title("{} & {}".format(data_name, model_name))
        plt.xticks(select_size_ratio_arr)
        plt.xlabel("Sel%")  # "percentage of test cases (%)"
        plt.ylabel("Imp%")  # "percentage of retraining accuracy improved (%)"
        plt.legend()
        for fig_suffix in self.fig_suffixes:
            plt.savefig(output_dir + "/{}.{}".format(fig_name, fig_suffix), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    RQ2Fig().plot_figs()
