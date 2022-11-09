import os
import random
import time
from collections import defaultdict

import keras
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from tqdm import tqdm

from pt import TriProCover
from utils import model_conf
import numpy as np
import pandas as pd

from gen_data.Adv import MyAdv
from gen_data.CifarDau import CifarDau
from gen_data.FashionDau import FashionDau
from gen_data.MnistDau import MnistDau
import matplotlib.pyplot as plt

from gen_data.SvhnDau import SvhnDau
from nc_coverage.neural_cov import CovRank

plt.switch_backend('agg')
from utils.utils import shuffle_data, add_df, num_to_str
from keras import backend as K


##### 重训练
def retrain_detail_all(x_s, y_s, X_train, Y_train, x_val_dict, y_val_dict, model_path, nb_classes,
                       verbose=1):
    if x_s is None and y_s is None:
        Ya_train = Y_train
        Xa_train = X_train
    else:
        # 1 . 合并训练集
        Ya_train = np.concatenate([Y_train, y_s])
        Xa_train = np.concatenate([X_train, x_s])
    # 2. hot
    Ya_train_vec = keras.utils.np_utils.to_categorical(Ya_train, nb_classes)
    # Y_test_vec = keras.utils.np_utils.to_categorical(Y_test, nb_classes)

    # 2. 加载模型
    ori_model = load_model(model_path)
    # 在 测试集上的精度  准确性

    # 验证集
    x_val_arr = []
    y_val_arr = []
    val_base_dict = {}
    for op, x_val in x_val_dict.items():
        y_val = y_val_dict[op]
        y_val_vec = keras.utils.np_utils.to_categorical(y_val, nb_classes)
        x_val_arr.append(x_val)
        y_val_arr.append(y_val)
        acc_si_val_ori = ori_model.evaluate(x_val, y_val_vec, verbose=0)[1]
        val_base_dict[op] = acc_si_val_ori

    X_val = np.concatenate(x_val_arr, axis=0)
    Y_val = np.concatenate(y_val_arr, axis=0)
    Y_val_vec = keras.utils.np_utils.to_categorical(Y_val, nb_classes)

    # 在 验证集上的精度  泛化鲁邦性
    acc_base_val = ori_model.evaluate(X_val, Y_val_vec, verbose=0)[1]
    sss = time.time()
    trained_model = retrain_model(ori_model, Xa_train, Ya_train_vec, X_val, Y_val_vec, "cov", 0,
                                  verbose=verbose)
    eee = time.time()
    acc_si_val = trained_model.evaluate(X_val, Y_val_vec, verbose=0)[1]

    acc_imp_val = acc_si_val - acc_base_val
    val_si_dict = {}
    val_si_dict["all"] = acc_imp_val
    for op, x_val in x_val_dict.items():
        y_val = y_val_dict[op]
        y_val_vec = keras.utils.np_utils.to_categorical(y_val, nb_classes)
        acc_si_val_op = trained_model.evaluate(x_val, y_val_vec, verbose=0)[1]
        val_si_dict[op] = acc_si_val_op - val_base_dict[op]

    print("val acc", acc_base_val, acc_si_val, "diff:", format(acc_imp_val, ".3f"))
    K.clear_session()  # 每次重训练后都清缓存
    return val_si_dict, eee - sss


def retrain_model(ori_model, x_si, y_si_vector, Xa_test, Ya_test_vec, smaple_method, idx=0, verbose=1):
    temp_path = model_conf.get_temp_model_path(data_name, model_name, smaple_method)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    new_model_name = exp_name + str(idx) + "_.hdf5"
    filepath = "{}/{}".format(temp_path, new_model_name)
    trained_model = train_model(ori_model, filepath, x_si, y_si_vector, Xa_test, Ya_test_vec, verbose=verbose)
    return trained_model


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=7, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


###### ix
# 1.实现cov idx当cam不够时,使用随机补
def get_retrain_idx(cam_path, ctm_path, select_size, Xa_train_size):
    ctm_idx, cam_idx = None, None
    if ctm_path is not None and os.path.exists(ctm_path):
        # print("ctm exits")
        ctm_ps_arr = np.load(ctm_path)
        ctm_idx = ctm_ps_arr[:select_size]
    if cam_path is not None and os.path.exists(cam_path):
        # print("cam exits")
        cam_ps_arr = np.load(cam_path)
        if len(cam_ps_arr) >= select_size:
            cam_idx = cam_ps_arr[:select_size]
        else:  # cam不够,随机机选补充
            diff_size = select_size - len(cam_ps_arr)  # cam全放里不够,剩下的随机补
            diff_idx = list(set(list(range(Xa_train_size))) - set(cam_ps_arr))  # 剩下的测试用例
            random.seed(0)
            idx = random.sample(diff_idx, diff_size)  # 随机选差的数量
            cam_idx = list(cam_ps_arr) + list(idx)  # 将cam的序列放前面,随机的放后面,
    if cam_idx is not None and len(cam_idx) != select_size:
        raise ValueError("cam选出的用例数与预期不一致!")

    if ctm_idx is not None and len(ctm_idx) != select_size:
        raise ValueError("ctm选出的用例数与预期不一致!")
    return cam_idx, ctm_idx


# 2.获得cov指标的重训练下标顺序,
def get_cov_retrain_idx(name, select_size, Xa_train_len, ps_path, prefixx=""):  #
    idx_data = {}
    ctm_path = ps_path + "{}_{}_rank_list{}.npy".format(name, "ctm", prefixx)
    cam_path = ps_path + "{}_{}_rank_list{}.npy".format(name, "cam", prefixx)
    cam_idx, ctm_idx = get_retrain_idx(cam_path, ctm_path, select_size, Xa_train_len)
    idx_arr = [cam_idx, ctm_idx]
    prefix_arr = ['cam', "ctm"]
    for prefix, idx in zip(prefix_arr, idx_arr):
        if idx is None:
            continue
        idx_data[name + "_" + prefix] = idx
    return idx_data


# 获得space的重训练下标顺序,
def get_space_retrain_idx(name, select_size, Xa_train_len, ps_path, prefixx=""):  #
    idx_data = {}
    ctm_path = None
    cam_path = ps_path + "{}_{}_rank_list{}.npy".format(name, "cam", prefixx)
    cam_idx, _ = get_retrain_idx(cam_path, ctm_path, select_size, Xa_train_len)

    idx_arr = [cam_idx]
    prefix_arr = ['cam']
    for prefix, idx in zip(prefix_arr, idx_arr):
        if idx is None:
            print(name, "idx", "is None")
            continue
        idx_data[name + "_" + prefix] = idx
    return idx_data


# random的重训练下标顺序,
def get_random_retrain_idx(name, select_size, len_x, seed=None, ):
    method = "ALL"
    if seed is not None:
        np.random.seed(seed)
    shuffle_indices = np.random.permutation(len_x)
    shuffle_indices_select = shuffle_indices[:select_size]
    res = {
        "{}_{}".format(name, method): shuffle_indices_select
    }
    return res


##### csv
# 获得返回值的map
def get_ps_csv_data():
    ps_collection_data = {
        "name": None,
        "rate": None,
        "t_collection": None,
        "cam_t_selection": None,
        "cam_max": None,
        "ctm_t_selection": None,
    }
    return ps_collection_data


def get_retrain_csv_data(name, method, imp_dict, time):
    csv_data = {
        "name": name,
        "method": method,
        "time": time,
    }
    for op, x_val in imp_dict.items():
        csv_data[op] = imp_dict[op]
    return csv_data


#####cover
def get_cov_initer(X_train, Y_train, data_name, model_name):
    from nc_coverage.neural_cov import CovInit
    params = {
        "data_name": data_name,
        "model_name": model_name
    }
    cov_initer = CovInit(X_train, Y_train, params)
    return cov_initer


def get_cov_name_and_func(cov_name_list, model_path, cov_initer, x_s, y_s):
    cov_ranker = CovRank(cov_initer, model_path, x_s, y_s)
    func_list = []
    name_func_map = {
        "NAC": cov_ranker.cal_nac_cov,
        "NBC": cov_ranker.cal_nbc_cov,
        "SNAC": cov_ranker.cal_snac_cov,
        "TKNC": cov_ranker.cal_tknc_cov,
        "LSC": cov_ranker.cal_lsc_cov,
        "KMNC": cov_ranker.cal_kmnc_cov
    }

    for cov_name in cov_name_list:
        func_list.append(name_func_map[cov_name])
    return func_list


##### ps
def prepare_ps(tripro_cover, cov_name_list, is_cov, is_space, base_path, model_path, cov_initer, x_s, y_s, nb_classes,
               prefix=""):
    print("prepareing ps...")
    df = None
    if is_cov:
        func_list = get_cov_name_and_func(cov_name_list, model_path, cov_initer, x_s, y_s)
        df = prepare_cov_ps(base_path, cov_name_list, func_list, df=df, prefix=prefix)
    if is_space:
        df = prepare_space_ps(tripro_cover, nb_classes, base_path, model_path, x_s, y_s, df=df, prefix=prefix)
    return df


def prepare_space_ps(tripro_cover: TriProCover, nb_classes, base_path, model_path, x_s, y_s,
                     df=None, prefix="", use_shuffle=False, seed=0):
    csv_data = get_ps_csv_data()
    ori_model = load_model(model_path)
    if use_shuffle:
        np.random.seed(seed)
        ix = np.random.permutation(len(x_s))
        x_s, y_s = y_s[ix], y_s[ix]
        print("space have shuffle data")
    pretreatment_arr = [None]  # "SPACE_CTM"
    method_arr = ["cam"]  # "ctm1cam"
    name = "DeepSpace"
    for pretreatment, method in zip(pretreatment_arr, method_arr):
        s = time.time()
        x_s_prob_matrix = ori_model.predict(x_s)
        x_test_space, y_test_space, ix_arr, cov_rate, max_cov_num = tripro_cover.rank_greedy(x_s, x_s_prob_matrix,
                                                                                             y_s,
                                                                                             nb_classes,
                                                                                             deep_num)

        e = time.time()
        csv_data["name"] = name + "_" + method
        csv_data["t_collection"] = 0
        csv_data["cam_t_selection"] = e - s
        csv_data["cam_max"] = max_cov_num
        csv_data["rate"] = cov_rate
        df = add_df(df, csv_data)
        save_path = base_path + "/ps_data/{}_{}_rank_list{}.npy".format(name, method, prefix)
        np.save(save_path, ix_arr)
    return df


def prepare_cov_ps(base_path, cov_name_list, func_list, df=None, prefix=""):
    csv_data = get_ps_csv_data()
    for name, func in zip(cov_name_list, func_list):
        rate, t_collection, rank_lst, t_selection_cam, rank_lst2, t_selection_ctm = func()
        csv_data["name"] = name
        csv_data["t_collection"] = t_collection
        if rank_lst is not None:
            save_path = base_path + "/ps_data/{}_{}_rank_list{}.npy".format(name, "cam", prefix)
            np.save(save_path, rank_lst)
            csv_data["cam_t_selection"] = t_selection_cam
            csv_data["cam_max"] = len(rank_lst)
            csv_data["rate"] = rate
        if rank_lst2 is not None:
            save_path = base_path + "/ps_data/{}_{}_rank_list{}.npy".format(name, "ctm", prefix)
            np.save(save_path, rank_lst2)
            csv_data["ctm_t_selection"] = t_selection_ctm
        df = add_df(df, csv_data)
    return df


##### exp
def mk_exp_dir(data_name, model_name):
    # 6.进行试验
    ## 6.1 创建文件夹并储存该次参数文件
    base_path = "./result"
    pair_name = model_conf.get_pair_name(data_name, model_name)
    dir_name = exp_name + "_" + pair_name
    txt_name = pair_name + ".txt"
    base_path = base_path + "/" + dir_name
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    txt_path = base_path + "/" + txt_name
    # param2txt(txt_path, json.dumps(params, indent=1))
    return base_path


def get_dau(data_name):
    if data_name == model_conf.mnist:
        return MnistDau()
    if data_name == model_conf.fashion:
        return FashionDau()
    if data_name == model_conf.svhn:
        return SvhnDau()
    if data_name == model_conf.cifar10:
        return CifarDau()


def get_dau_data(x_test, y_test, dau, dau_name_arr, ratio=0.5, use_shuffle=False):
    x_test_arr = []
    y_test_arr = []

    x_val_dict = {}
    y_val_dict = {}
    # 添加原始的
    num = int(len(x_test) * ratio)
    x_test_arr.append(x_test[:num])
    y_test_arr.append(y_test[:num])
    x_val_dict["ORI"] = x_test[num:]
    y_val_dict["ORI"] = y_test[num:]

    # 添加扩增的
    for dau_op_name in dau_name_arr:
        # print(dau_op_name)
        x, y = dau.load_dau_data(dau_op_name, use_norm=True)
        if use_shuffle:
            x, y = shuffle_data(x, y, 0)
        num = int(len(x) * ratio)
        x_test_arr.append(x[:num])
        y_test_arr.append(y[:num])

        x_val_dict[dau_op_name] = x[num:]
        y_val_dict[dau_op_name] = y[num:]

    x_dau_test = np.concatenate(x_test_arr, axis=0)
    y_dau_test = np.concatenate(y_test_arr, axis=0)

    return x_dau_test, y_dau_test, x_val_dict, y_val_dict


def get_adv_data(x_test, y_test, attack_name_arr, data_name, model_name, ratio=0.2):
    x_test_arr = []
    y_test_arr = []
    x_test_arr.append(x_test)
    y_test_arr.append(y_test)

    x_val_dict = {}
    y_val_dict = {}
    adv = MyAdv()
    for attack_name in attack_name_arr:
        x, y = adv.load_adv_data(attack_name, data_name, model_name)
        x, y = shuffle_data(x, y, seed=1)
        num = int(len(x_test) * ratio)
        x_test_arr.append(x[:num])
        y_test_arr.append(y[:num])
        x_val_dict[attack_name] = x[num:]
        y_val_dict[attack_name] = y[num:]
        print(attack_name, len(x), len(x_test_arr[-1]), len(x_val_dict[attack_name]))
    x_adv_test = np.concatenate(x_test_arr, axis=0)
    y_adv_test = np.concatenate(y_test_arr, axis=0)
    return x_adv_test, y_adv_test, x_val_dict, y_val_dict


def exec(model_name, data_name):
    # 实验
    base_path = mk_exp_dir(data_name, model_name)
    exp(model_name, data_name, base_path)
    # exp_retrain_cam_max(model_name, data_name, base_path) #Retrain cam_max tests -->RQ2 table
    K.clear_session()


def exp(model_name, data_name, base_path, ):
    is_prepare_cov = True
    is_prepare_space = True
    is_retrain_cov = True
    is_retrain_space = True
    is_retrain_random = True
    verbose = 0
    is_retrain_all = True
    cov_name_list = ["NBC", "SNAC", "NAC", "TKNC", "LSC"]
    dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'NS', 'BL', 'SR']

    print(dau_name_arr)
    attack_name_arr = ["bim", "pgd", "jsma", "ead", "fgsm"]  # "ead", "ead",
    dau = get_dau(data_name)
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    ps_path = "{}/ps_data/".format(base_path)
    os.makedirs(ps_path, exist_ok=True)
    ps_csv_dir = "{}/priority_sequence".format(base_path)
    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    tripro_cover = TriProCover()
    prefix = "_" + mode
    ps_csv_path = "{}_{}.csv".format(ps_csv_dir, mode)
    if mode == "dau":
        x_select, y_select, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                  use_shuffle=True)
    elif mode == "adv":
        x_select, y_select, x_val_dict, y_val_dict = get_adv_data(x_test, y_test, attack_name_arr, data_name,
                                                                  model_name)
    else:
        raise ValueError("end with no exp")

    x_select, y_select = shuffle_data(x_select, y_select, 0)  # 加载的数据都是有顺序的

    select_size_arr = []
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len(x_select) * select_size_ratio)
        select_size_arr.append(select_size)

    print("mode :{}".format(mode), "size:", select_size_arr, "keys:", x_val_dict.keys())
    model_path = model_conf.get_model_path(data_name, model_name)
    cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)
    nb_classes = model_conf.fig_nb_classes
    df_ps = prepare_ps(tripro_cover, cov_name_list, is_prepare_cov, is_prepare_space,
                       base_path, model_path, cov_initer,
                       x_select, y_select,
                       nb_classes, prefix=prefix)
    if df_ps is not None:
        df_ps.to_csv(ps_csv_path, index=False)

    res_all = defaultdict(dict)
    for select_size in select_size_arr:
        print(select_size, is_retrain_cov, is_retrain_space, is_retrain_random)
        df = None
        csv_path = os.path.join(base_path, "res_{}_{}.csv").format(mode, select_size)
        idx_data = get_idx_data(is_retrain_cov, is_retrain_space, is_retrain_random, cov_name_list, len(x_select),
                                select_size, ps_path,
                                prefix)
        for k, idx in tqdm(idx_data.items()):
            method = str(k).split("_")[-1]
            name = str(k).split("_")[0]
            x_s, y_s = x_select[idx], y_select[idx]
            imp_dict, retrain_time = retrain_detail_all(x_s, y_s, x_train, y_train,
                                                        x_val_dict, y_val_dict,
                                                        model_path, nb_classes,
                                                        verbose=verbose)
            res_all[k]["acc_{}".format(select_size)] = imp_dict["all"]
            cov_trained_csv_data = get_retrain_csv_data(name, method, imp_dict, retrain_time)
            df = add_df(df, cov_trained_csv_data)
            df.to_csv(csv_path, index=False)
        print("over", csv_path)
    #### 添加一个 retrain all
    if is_retrain_all:
        imp_dict, retrain_time = retrain_detail_all(x_select, y_select, x_train, y_train,
                                                    x_val_dict, y_val_dict,
                                                    model_path, nb_classes,
                                                    verbose=verbose)
        total_acc = imp_dict["all"]
    else:
        total_acc = 1
    df_all = None
    csv_path_all = os.path.join(base_path, "res{}_all.csv").format(prefix)

    for k, k_csv_data in res_all.items():
        k_csv_data["name"] = k
        k_csv_data["acc_total"] = total_acc
        for select_size in select_size_arr:
            k_csv_data["p_acc_{}".format(select_size)] = \
                num_to_str(k_csv_data["acc_{}".format(select_size)] / total_acc, 5)
        df_all = add_df(df_all, k_csv_data)
        df_all.to_csv(csv_path_all, index=False)
        print("over", csv_path_all)
    plot_line_figs2(select_size_arr, csv_path_all, base_path, prefix, select_size_ratio_arr)


def get_idx_data(is_retrain_cov, is_retrain_space, is_retrain_random, cov_name_list, len_x_select, select_size, ps_path,
                 prefix):
    idx_data = {}
    if is_retrain_cov:
        # 1. cov
        for name in cov_name_list:
            temp_idx_data = get_cov_retrain_idx(name, select_size, len_x_select,
                                                ps_path, prefixx=prefix)
            idx_data = dict(idx_data, **temp_idx_data)
    if is_retrain_space:
        name = "DeepSpace"
        # method = "cam"
        temp_idx_data = get_space_retrain_idx(name, select_size, len_x_select, ps_path, prefixx=prefix)
        idx_data = dict(idx_data, **temp_idx_data)
    if is_retrain_random:
        # 3.random
        name = "Random"
        temp_idx_data = get_random_retrain_idx(name, select_size, len_x_select, seed=None)
        idx_data = dict(idx_data, **temp_idx_data)
    return idx_data


def cal_cov(k, cov_initer, x_s, y_s):
    from nc_coverage import metrics
    input_layer, layers = cov_initer.input_layer, cov_initer.layers
    if "NAC" in k:
        nac = metrics.nac(x_s, input_layer, layers, t=0.75)
        rate = nac.fit()
    elif "NBC" in k:
        nbc = cov_initer.get_nbc(std=0)
        rate = nbc.fit(x_s, use_lower=True)
    elif "SNAC" in k:
        snac = cov_initer.get_nbc(std=0)
        rate = snac.fit(x_s, use_lower=False)
    elif "KMNC" in k:
        kmnc = cov_initer.get_kmnc(k_bins=1000, time_limit=3600, max_select_size=None)
        rate = kmnc.fit(x_s)
    elif "TKNC" in k:
        tknc = metrics.tknc(x_s, input_layer, layers, k=1)
        rate = tknc.fit(list(range(len(x_s))))
    elif "LSC" in k:
        lsc = cov_initer.get_lsc(k_bins=1000, index=-1, u=100)
        rate = lsc.fit(x_s, y_s)
    else:
        raise ValueError()
    return rate


def exp_retrain_cam_max(model_name, data_name, base_path):
    cov_name_list = ["NBC", "SNAC", "NAC", "TKNC", "LSC"]
    dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'NS', 'BL', 'SR']
    attack_name_arr = ["bim", "pgd", "jsma", "ead", "fgsm"]
    dau = get_dau(data_name)
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    ps_path = "{}/ps_data/".format(base_path)
    os.makedirs(ps_path, exist_ok=True)
    ps_csv_dir = "{}/priority_sequence".format(base_path)
    prefix = "_" + mode
    ps_csv_path = "{}_{}.csv".format(ps_csv_dir, mode)
    if mode == "dau":
        x_select, y_select, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                  use_shuffle=True)
    elif mode == "adv":
        x_select, y_select, x_val_dict, y_val_dict = get_adv_data(x_test, y_test, attack_name_arr, data_name,
                                                                  model_name)
    else:
        raise ValueError("end with no exp")

    x_select, y_select = shuffle_data(x_select, y_select, 0)  # 加载的数据都是有顺序的

    model_path = model_conf.get_model_path(data_name, model_name)
    nb_classes = model_conf.fig_nb_classes

    idx_data = {}
    # space
    ps_cov_path = os.path.join(ps_path, "{}_{}_rank_list{}.npy".format("DeepSpace", "cam", prefix))
    ps_idx_arr = np.load(ps_cov_path)
    idx_data["DeepSpace" + "_" + "cam"] = ps_idx_arr

    # cov
    for cov_name in cov_name_list:
        ps_cov_path = os.path.join(ps_path, "{}_{}_rank_list{}.npy".format(cov_name, "cam", prefix))
        ps_idx_arr = np.load(ps_cov_path)
        idx_data[cov_name + "_" + "cam"] = ps_idx_arr

    df = None
    csv_path_all = os.path.join(base_path, "res{}_all.csv").format(prefix)
    df_all = pd.read_csv(csv_path_all)
    total = df_all["acc_total"][0]
    csv_path = os.path.join(base_path, "res_{}_{}.csv").format(mode, "max")

    df_ps = pd.read_csv(ps_csv_path)
    for k, idx in tqdm(idx_data.items()):
        x_s, y_s = x_select[idx], y_select[idx]
        res = {}
        name = k.split("_")[0]
        method = k.split("_")[1]
        res['name'] = name
        res['method'] = method

        if "DeepSpace" in k:
            df_ps_row = df_ps[df_ps['name'] == k]
            res["time"] = df_ps_row["t_collection"].values[0] + df_ps_row["cam_t_selection"].values[0]
            res["rate"] = df_ps_row["rate"].values[0]
        else:
            df_ps_row = df_ps[df_ps['name'] == name]
            res["time"] = df_ps_row["t_collection"].values[0] + df_ps_row["cam_t_selection"].values[0]
            cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)
            res["rate"] = cal_cov(k, cov_initer, x_s, y_s)
            del cov_initer
            K.clear_session()
        max_cov_size = df_ps_row["cam_max"].values[0]
        print(max_cov_size)
        res["cam_max"] = max_cov_size

        print("*************", k, len(idx))
        imp_dict, retrain_time = retrain_detail_all(x_s, y_s, x_train, y_train,
                                                    x_val_dict, y_val_dict,
                                                    model_path, nb_classes,
                                                    verbose=0)
        assert max_cov_size == len(x_s)

        acc_max = imp_dict["all"]
        res['all'] = acc_max
        res['acc_total'] = total
        res["p_acc_max"] = num_to_str(acc_max / total, 5)
        df = add_df(df, res)
        df.to_csv(csv_path, index=False)
    print("over", csv_path)


def plot_line_figs2(select_size_arr, csv_path, base_path, prefix, select_size_ratio_arr):
    name_arr = []
    select_arr = select_size_arr
    for select_size in select_arr:
        name_arr.append("p_acc_{}".format(select_size))
    df = pd.read_csv(csv_path)

    pair_name = model_conf.get_pair_name(data_name, model_name)
    for index, row in df.iterrows():
        print(row)
        name = row["name"]
        res_arr = [row[x] for x in name_arr]
        lb = str(name).split("_")[0]
        if "DeepSpace" in name:
            plt.plot(select_size_ratio_arr, res_arr, label=lb, color="crimson", marker="o")
        else:
            plt.plot(select_size_ratio_arr, res_arr, label=lb, marker="x", alpha=0.5)
    plt.xticks(select_size_ratio_arr)
    plt.legend()
    plt.savefig(base_path + "/{}{}_all.png".format(pair_name, prefix))
    plt.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mode_arr = ["dau", "adv"]
    for mode in mode_arr:
        if mode == "dau":
            exp_name = "selection_dau"
        else:
            exp_name = "selection_adv"
        deep_num = 4

        for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
            for model_name in model_name_arr:
                exec(model_name, data_name)
        # ####### example
        # model_name = model_conf.LeNet1
        # data_name = model_conf.mnist
        # exec(model_name, data_name)
