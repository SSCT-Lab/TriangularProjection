import os

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import model_conf
import seaborn as sns
import pandas as pd
from gen_table.my_fig import MyFig
from gen_table.my_table import MyTable
import numpy as np


class RQ1Figs(MyFig):
    def __init__(self, ):
        self.tab = RQ1Table()
        self.fig_type_arr = ["line", "bar", "box"]

        super().__init__(self.tab)

    def init_dirs(self):
        for mode in self.tab.mode_arr:
            for fig_type in self.fig_type_arr:
                output_base_path = self.get_output_fig_base_path(mode, fig_type)
                os.makedirs(output_base_path, exist_ok=True)

    def get_output_fig_base_path(self, mode, fig_type):
        output_base_path = os.path.join(self.output_fig_dir_path, self.tab.prefix, mode, fig_type)
        return output_base_path

    def plot_figs(self):
        k_arr = self.tab.get_df_columns_arr()
        for mode in tqdm(self.tab.mode_arr):
            for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
                for model_name in model_name_arr:
                    dir_path = self.tab.get_dir_path(mode, data_name, model_name)
                    if not os.path.exists(dir_path):
                        print(dir_path, "not exist")
                        continue
                    csv_path = os.path.join(dir_path, self.tab.csv_file)
                    # print("====", csv_path)
                    self.plot_line_figs(k_arr, data_name, model_name, mode, csv_path)

                    self.plot_bar_figs(k_arr, data_name, model_name, mode, csv_path)

                    self.plot_box_figs(k_arr, data_name, model_name, mode, csv_path)

    def plot_line_figs(self, k_arr, data_name, model_name, mode, csv_path):
        fig_name = self.get_figs_name(data_name, model_name)
        fig_type = self.fig_type_arr[0]
        output_dir = self.get_output_fig_base_path(mode, fig_type)
        self._plot_line_figs_detail(k_arr, csv_path, output_dir, fig_name)

    def plot_bar_figs(self, k_arr, data_name, model_name, mode, csv_path):
        fig_type = self.fig_type_arr[1]
        fig_name = self.get_figs_name(data_name, model_name)
        output_dir = self.get_output_fig_base_path(mode, fig_type)
        self._plot_bar_figs_detail(k_arr, csv_path, output_dir, fig_name)

    def plot_box_figs(self, k_arr, data_name, model_name, mode, csv_path):
        fig_type = self.fig_type_arr[2]
        fig_dir = self.get_figs_name(data_name, model_name)
        output_dir = self.get_output_fig_base_path(mode, fig_type)
        self._plot_box_figs_detail(k_arr, csv_path, output_dir, fig_dir)

    def _plot_box_figs_detail(self, k_arr, df_csv_path, output_dir, fig_dir):
        output_path = os.path.join(output_dir, fig_dir)
        os.makedirs(output_path, exist_ok=True)
        df = pd.read_csv(df_csv_path)
        for k in k_arr:
            if k in df.columns:
                p2 = sns.boxplot(x=df["comb_num"], y=df[k])
                p2 = sns.swarmplot(x=df["comb_num"], y=df[k], color=".25")
                p_res, _ = pearsonr(df["comb_num"], df[k])
                p_res = self.tab.num_to_str(p_res, 5)
                lb = self.get_lb_dict()[k]
                plt.title(lb + "_" + p_res)
                # plt.show()
                for fig_suffix in self.fig_suffixes:
                    plt.savefig(output_path + "/{}.{}".format(k, fig_suffix), bbox_inches='tight')
                plt.close()

    def _plot_line_figs_detail(self, k_arr, df_csv_path, output_dir, fig_name):

        df_ori = pd.read_csv(df_csv_path)
        # print(df_ori)
        df = df_ori.groupby('comb_num', as_index=False).mean()

        for k in k_arr:
            # print(k, df.columns)
            if k in df.columns:
                res_arr = df[k].copy()
                res_arr /= res_arr[0]
                cl = self.get_color_dict()[k]
                lb = self.get_lb_dict()[k]
                if "sp_c_4" in k:
                    plt.plot(df["comb_num"], res_arr, label=lb, color=cl, marker="o")
                else:
                    plt.plot(df["comb_num"], res_arr, label=lb, color=cl, marker="x", alpha=0.8)
        plt.legend()
        for fig_suffix in self.fig_suffixes:
            fig_path = os.path.join(output_dir, "{}.{}".format(fig_name, fig_suffix))
            plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # plt.savefig(fig_path)
        # plt.close()

    def _plot_bar_figs_detail(self, k_arr, df_csv_path, output_dir, fig_name):
        df_ori = pd.read_csv(df_csv_path)
        df_mean = df_ori.groupby('comb_num', as_index=False).mean()
        df_max = df_ori.groupby('comb_num', as_index=False).max()
        df_min = df_ori.groupby('comb_num', as_index=False).min()

        for k in k_arr:
            if k in df_mean.columns:
                mean_arr = df_mean[k].copy()
                mean_arr /= mean_arr[0]
                max_arr = df_max[k].copy()
                max_arr /= max_arr[0]
                min_arr = df_min[k].copy()
                min_arr /= min_arr[0]
                max_err = np.array(max_arr) - np.array(mean_arr)
                min_err = np.array(mean_arr) - np.array(min_arr)
                cl = self.get_color_dict()[k]
                lb = self.get_lb_dict()[k]
                if "sp_c_4" in k:
                    plt.errorbar(df_mean["comb_num"], mean_arr, label=lb, color=cl, marker="o",
                                 yerr=[min_err, max_err])
                else:
                    plt.errorbar(df_mean["comb_num"], mean_arr, label=lb, color=cl, alpha=0.5, marker="x",
                                 yerr=[min_err, max_err])
        plt.legend()
        for fig_suffix in self.fig_suffixes:
            plt.savefig(output_dir + "/{}.{}".format(fig_name, fig_suffix), bbox_inches='tight')
        plt.close()


class RQ1Table(MyTable):

    def __init__(self):
        self.csv_file = "res.csv"
        self.prefix = "correlation"
        super().__init__(self.prefix)

    def get_output_csv_path(self, mode):
        output_csv_path = os.path.join(self.output_tab_dir_path, "{}_{}.csv".format(self.prefix, mode))
        print("output_csv_path", output_csv_path)
        return output_csv_path

    def get_table_data(self):
        k_arr = self.get_df_columns_arr()

        for mode in self.mode_arr:
            df_res = self.get_table_frame()
            output_csv_path = self.get_output_csv_path(mode)
            for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
                for model_name in model_name_arr:
                    dir_path = self.get_dir_path(mode, data_name, model_name)
                    if not os.path.exists(dir_path):
                        print(dir_path, "not exist")
                        continue
                    csv_path = os.path.join(dir_path, self.csv_file)
                    df_data = pd.read_csv(csv_path)
                    df_data = df_data[1:]
                    # print(df_data)
                    for k in k_arr:
                        if k in df_data.columns:
                            p_res, r_res = spearmanr(df_data["comb_num"], df_data[k])
                            # print(p_res)
                            p_res = self.num_to_str(p_res, 3)
                            r_res = self.num_to_str(r_res, 3)
                            value = "({}/{})".format(p_res, r_res)
                            table_index = self.get_tab_index_by_df_columns(k)
                            table_column = self.get_table_column(data_name, model_name)
                            # print(table_index, table_column)
                            df_res.loc[table_index, table_column] = value
            df_res.to_csv(output_csv_path)


if __name__ == '__main__':
    RQ1Table().get_table_data()
    RQ1Figs().plot_figs()
