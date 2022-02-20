import os
import time

from keras.engine.saving import load_model
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import model_conf
import numpy as np
import pandas as pd

from gen_data.Adv import MyAdv
from gen_data.CifarDau import CifarDau
from gen_data.FashionDau import FashionDau
from gen_data.MnistDau import MnistDau
import matplotlib.pyplot as plt

from gen_data.SvhnDau import SvhnDau
from tri_projection.TriangularProjection import TriProCover

plt.switch_backend('agg')
from utils.utils import shuffle_data, add_df, num_to_str
from keras import backend as K


def get_cov_exp_data(x_s, y_s, cov_initer, suffix=""):
    from nc_coverage import metrics
    csv_data = {}
    cov_nac, cov_nbc, cov_snac, cov_kmnc, cov_tknc, cov_lsc, cov_dsc = None, None, None, None, None, None, None,

    input_layer = cov_initer.get_input_layer()
    layers = cov_initer.get_layers()

    ss = time.time()
    nac = metrics.nac(x_s, input_layer, layers, t=0.75)
    cov_nac = nac.fit()
    ee = time.time()
    csv_data["nac_time{}".format(suffix)] = ee - ss
    #
    sss = time.time()
    nbc = cov_initer.get_nbc()
    eee = time.time()
    base_time = eee - sss
    ss = time.time()
    cov_nbc = nbc.fit(x_s, use_lower=True)
    ee = time.time()
    csv_data["nbc_time{}".format(suffix)] = ee - ss + base_time
    ss = time.time()
    cov_snac = nbc.fit(x_s, use_lower=False)
    ee = time.time()
    csv_data["snac_time{}".format(suffix)] = ee - ss + base_time
    #
    ss = time.time()
    kmnc = cov_initer.get_kmnc()
    cov_kmnc = kmnc.fit(x_s)
    ee = time.time()
    csv_data["kmnc_time{}".format(suffix)] = ee - ss

    ss = time.time()
    tknc = metrics.tknc(x_s, input_layer, layers, k=1)
    cov_tknc = tknc.fit(list(range(len(x_s))))
    ee = time.time()
    csv_data["tknc_time{}".format(suffix)] = ee - ss

    ss = time.time()
    lsc = cov_initer.get_lsc(k_bins=1000, index=-1)
    cov_lsc = lsc.fit(x_s, y_s)
    ee = time.time()
    csv_data["lsc_time{}".format(suffix)] = ee - ss

    csv_data["cov_nac{}".format(suffix)] = cov_nac
    csv_data["cov_nbc{}".format(suffix)] = cov_nbc
    csv_data["cov_snac{}".format(suffix)] = cov_snac
    csv_data["cov_tknc{}".format(suffix)] = cov_tknc
    csv_data["cov_kmnc{}".format(suffix)] = cov_kmnc
    csv_data["cov_lsc{}".format(suffix)] = cov_lsc
    return csv_data


def get_cov_initer(X_train, Y_train, data_name, model_name):
    from nc_coverage.neural_cov import CovInit
    params = {
        "data_name": data_name,
        "model_name": model_name
    }
    cov_initer = CovInit(X_train, Y_train, params)
    return cov_initer


def get_dau(data_name):
    if data_name == model_conf.mnist:
        return MnistDau()
    if data_name == model_conf.fashion:
        return FashionDau()
    if data_name == model_conf.svhn:
        return SvhnDau()
    if data_name == model_conf.cifar10:
        return CifarDau()


def exp_detail(deep_num, tripro_cover: TriProCover, x_select, y_select, nb_classes, ori_model, csv_data, use_space,
               use_cov,
               cov_initer):
    # print(len(x_select))
    if use_space:
        s = time.time()
        sp_c_arr, sp_v_arr = tripro_cover.cal_triangle_cov(x_select, y_select, nb_classes, ori_model, deep_num,
                                                           by_deep_num=True)  # 把中间结果也记录一下
        e = time.time()
        sp_c_str_arr = [num_to_str(x, 5) for x in sp_c_arr]
        sp_data2 = {"sp_time": e - s}
        for i, sp_c_str in enumerate(sp_c_str_arr):
            sp_data2["sp_c_{}".format(i + 1)] = sp_c_str
        csv_data = dict(csv_data, **sp_data2)

    if use_cov:
        cov_data = get_cov_exp_data(x_select, y_select, cov_initer, suffix="")
        csv_data = dict(csv_data, **cov_data)
    del y_select
    del x_select
    return csv_data


def exp(model_name, data_name, base_path, ):
    deep_num = 4
    use_cov = True
    use_space = True
    sample_num = 5  # 2
    replace_ratio = 0.1
    dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'NS', 'BL', 'SR']
    attack_name_arr = ["bim", "pgd", "jsma", "ead", "fgsm"]

    # 加载模型
    model_path = model_conf.get_model_path(data_name, model_name)
    ori_model = load_model(model_path)

    # 扩增类
    dau = get_dau(data_name)
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)

    test_size = dau.test_size
    nb_classes = dau.nb_classes
    #
    cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)

    #####
    # 打印信息
    #####
    print("data_name", data_name)
    print("model_name", model_name)
    print("dau_path", dau.dau_dir)
    print("model_path", model_path)

    csv_path = base_path + "/" + "res.csv"  # 每组的储存路径
    df = None  # 每组的df
    print("res_path", csv_path)

    # 随机的算子
    np.random.seed(0)
    seed_list = np.random.permutation(10000)
    seed_idx = 0

    # 被随机替换的序列(原始)
    np.random.seed(1)
    seed_list1 = np.random.permutation(10000)
    seed_idx1 = 0

    # 随机替换的序列(扩增)
    np.random.seed(2)
    seed_list2 = np.random.permutation(10000)
    seed_idx2 = 0

    if mode == "dau":
        dop_name_arr = dau_name_arr
        adv = None
    else:
        dop_name_arr = attack_name_arr
        adv = MyAdv()
    print(mode, dop_name_arr)
    tripro_cover = TriProCover(is_save_profile=True, base_path=base_path, suffix="")
    # 每个组合的扩增算子
    # 共有i种组合
    for ii in range(0, len(dop_name_arr) + 1, 1):
        if ii == 0:
            # 记录原始精度
            x_select, y_select = x_test, y_test
            csv_data = {
                "comb_name": None,
                "comb_num": ii,
                "data_select_time": 0,
            }
            csv_data = exp_detail(deep_num, tripro_cover, x_select, y_select, nb_classes, ori_model, csv_data,
                                  use_space, use_cov,
                                  cov_initer)
            df = add_df(df, csv_data)
            df.to_csv(csv_path, index=False)
            del x_select
        else:
            # 每种组合里有i种算子
            print("=======================")
            comb = dop_name_arr[0:ii]  # 本次的扩增算子
            # print(ii, comb)
            for step in range(sample_num):  # 每个算子个数执行20次
                np.random.seed(seed_list[seed_idx])
                seed_idx += 1

                np.random.seed(seed_list1[seed_idx1])
                seed_idx1 += 1
                dau_idx = np.random.permutation(test_size)  # 随机选取替换序列

                x_arr = []
                y_arr = []
                start_idx = 0  # 起始索引
                # 添加扩增数据
                info_map = {}
                s = time.time()
                for dop_name in comb:
                    if mode == "dau":
                        x, y = dau.load_dau_data(dop_name, prefix="test", use_norm=True)
                        idx = dau_idx
                    else:
                        x, y = adv.load_adv_data(dop_name, data_name, model_name, use_cache=True)
                        np.random.seed(seed_list1[seed_idx1])
                        seed_idx1 += 1
                        idx = np.random.permutation(len(x))

                    x, y = shuffle_data(x, y, seed=seed_list2[seed_idx2])
                    seed_idx2 += 1
                    subsize = int(len(x) * replace_ratio)  # 每个扩增的子集大小
                    end_idx = start_idx + subsize  # 结束索引
                    print(dop_name, start_idx, end_idx, subsize)
                    sub_idx = idx[start_idx:end_idx]
                    x_s = x[sub_idx]
                    y_s = y[sub_idx]
                    x_arr.append(x_s)
                    y_arr.append(y_s)
                    info_map[dop_name] = dop_name
                    info_map[dop_name + "_start_idx"] = start_idx
                    info_map[dop_name + "_end_idx"] = end_idx
                    start_idx = end_idx
                # 添加原始数据
                x_ori = x_test[idx[start_idx:]]
                y_ori = y_test[idx[start_idx:]]
                x_arr.append(x_ori)
                y_arr.append(y_ori)
                x_select = np.concatenate(x_arr, axis=0)
                y_select = np.concatenate(y_arr, axis=0)
                # print("扩增数据: ", info_map, "原始数据", len(x_ori), start_idx, "总长度", len(x_select))
                x_select, y_select = shuffle_data(x_select, y_select, 0)  # 混洗数据
                del x_ori
                del x_arr
                e = time.time()
                # 1. 原始精度
                # acc = ori_model.evaluate(x_select, np_utils.to_categorical(y_select, nb_classes))[1]
                # print(acc)
                csv_data = {
                    "comb_name": "_".join(comb),
                    "comb_num": ii,
                    "data_select_time": e - s,
                }
                csv_data = exp_detail(deep_num, tripro_cover, x_select, y_select, nb_classes, ori_model, csv_data,
                                      use_space,
                                      use_cov, cov_initer, )
                df = add_df(df, csv_data)
                df.to_csv(csv_path, index=False)
                del x_select
            df.to_csv(csv_path, index=False)
    plot_box_figs(base_path)
    plot_line_figs(csv_path, base_path)
    plot_bar_figs(csv_path, base_path)

    if dau is not None:
        dau.clear_cache()
    if adv is not None:
        adv.clear_cache()
        del adv
    del dau


def plot_box_figs(base_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import seaborn as sns
    csv_path = base_path + "/" + "res.csv"
    df = pd.read_csv(csv_path)
    k_arr = ["cov_lsc", "cov_lsc2", "cov_lsc3", "cov_nbc", "cov_snac", "cov_nac", "cov_kmnc", "cov_tknc",
             "sp_c_1", "sp_c_2", "sp_c_3", "sp_c_4", ]
    for k in k_arr:
        if k in df.columns:
            p2 = sns.boxplot(x=df["comb_num"], y=df[k])
            p2 = sns.swarmplot(x=df["comb_num"], y=df[k], color=".25")
            p_res, _ = pearsonr(df["comb_num"], df[k])
            p_res = num_to_str(p_res, 5)
            plt.title(k + "_" + p_res)
            plt.savefig(base_path + "/{}.png".format(k))
            plt.close()


def plot_line_figs(csv_path, base_path):
    df_ori = pd.read_csv(csv_path)
    df = df_ori.groupby('comb_num', as_index=False).mean()
    df.to_csv(os.path.join(base_path, "res_line_point.csv"), index=False)
    k_arr = ["cov_lsc", "cov_nbc", "cov_snac", "cov_nac", "cov_kmnc",
             "sp_c_1", "sp_c_2", "sp_c_3", "sp_c_4", "cov_tknc"]
    fig_path = "{}/fig".format(base_path)
    os.makedirs(fig_path, exist_ok=True)
    for k in k_arr:
        if k in df_ori.columns:
            plt.plot(df["comb_num"], df[k])
            plt.title(k)
            plt.savefig(fig_path + "/{}.png".format(k))
            plt.close()

    k_arr = ["cov_lsc", "cov_lsc2", "cov_lsc3", "cov_nbc", "cov_snac", "cov_nac", "cov_kmnc",
             "sp_c_4", "cov_tknc"]

    pair_name = model_conf.get_pair_name(data_name, model_name)
    for k in k_arr:
        if k in df.columns:
            res_arr = df[k].copy()
            res_arr /= res_arr[0]
            if "sp_c_4" in k:
                plt.plot(df["comb_num"], res_arr, label="DeepSpace", color="crimson", marker="o")
            else:
                plt.plot(df["comb_num"], res_arr, label=k, alpha=0.5, marker="x")
    plt.legend()
    plt.savefig(base_path + "/{}_line.png".format(pair_name))
    plt.close()


def plot_bar_figs(csv_path, base_path):
    df_ori = pd.read_csv(csv_path)
    df_mean = df_ori.groupby('comb_num', as_index=False).mean()  # median()
    df_max = df_ori.groupby('comb_num', as_index=False).max()
    df_min = df_ori.groupby('comb_num', as_index=False).min()

    df_mean.to_csv(os.path.join(base_path, "res_line_point.csv"), index=False)

    k_arr = ["cov_lsc", "cov_lsc2", "cov_lsc3", "cov_nbc", "cov_snac", "cov_nac", "cov_kmnc",
             "sp_c_4", "cov_tknc"]
    pair_name = model_conf.get_pair_name(data_name, model_name)
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
            if "sp_c_4" in k:
                plt.errorbar(df_mean["comb_num"], mean_arr, label="DeepSpace", color="crimson", marker="o",
                             yerr=[min_err, max_err])
            else:
                plt.errorbar(df_mean["comb_num"], mean_arr, label=k, alpha=0.5, marker="x",
                             yerr=[min_err, max_err])
    plt.legend()
    plt.savefig(base_path + "/{}_bar.png".format(pair_name))
    plt.close()


def mk_exp_dir(data_name, model_name):
    # 6.进行试验
    ## 6.1 创建文件夹并储存该次参数文件
    base_path = "./result"
    pair_name = model_conf.get_pair_name(data_name, model_name)
    # dir_name = datetime.datetime.now().strftime("%m%d%H%M") + "_" + exp_name + "_" + pair_name
    dir_name = exp_name + "_" + pair_name
    txt_name = pair_name + ".txt"
    base_path = base_path + "/" + dir_name
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    txt_path = base_path + "/" + txt_name
    # param2txt(txt_path, json.dumps(params, indent=1))
    return base_path


def exec(model_name, data_name):
    # 实验
    base_path = mk_exp_dir(data_name, model_name)
    exp(model_name, data_name, base_path, )
    K.clear_session()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mode_arr = ["dau", "adv"]
    for mode in mode_arr:
        if mode == "dau":
            exp_name = "correlation_dau"
        else:
            exp_name = "correlation_adv"
        # 基本参数
        for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
            for model_name in model_name_arr:
                exec(model_name, data_name)

        # ####### example
        # model_name = model_conf.LeNet1
        # data_name = model_conf.mnist
        # exec(model_name, data_name)
