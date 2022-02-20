import os
import time

from keras.models import load_model
import numpy as np
import foolbox
from tqdm import tqdm

from utils import model_conf
import warnings

from keras import backend as K

from gen_data.CifarDau import CifarDau
from gen_data.FashionDau import FashionDau
from gen_data.MnistDau import MnistDau
from gen_data.SvhnDau import SvhnDau

warnings.filterwarnings("ignore")


# 获取图片参数
def get_image_wh(data_name):
    if data_name == model_conf.mnist or data_name == model_conf.fashion:
        w, h = 28, 28
    elif data_name == model_conf.cifar10 or data_name == model_conf.svhn:
        w, h = 32, 32
    else:
        raise ValueError()
    return w, h


def get_image_whc(data_name):
    if data_name == model_conf.mnist or data_name == model_conf.fashion:
        w, h, c = 28, 28, 1
    elif data_name == model_conf.cifar10 or data_name == model_conf.svhn:
        w, h, c = 32, 32, 3
    else:
        raise ValueError()
    return w, h, c


# 获取攻击方式
# 获取攻击方式
def get_attack(model_path, attack_name, ):
    model = load_model(model_path)
    foolmodel = foolbox.models.KerasModel(model, bounds=(0, 1), preprocessing=(0, 1))
    if attack_name == "bim":
        attack = foolbox.attacks.BIM(model=foolmodel, )
    elif attack_name == "ead":
        attack = foolbox.attacks.EADAttack(model=foolmodel, )
    elif attack_name == "pgd":
        attack = foolbox.attacks.RandomPGD(model=foolmodel, )
    elif attack_name == "deepfool":
        attack = foolbox.attacks.DeepFoolAttack(model=foolmodel, )
    elif attack_name == "newtonfool":
        attack = foolbox.attacks.NewtonFoolAttack(model=foolmodel, )
    # attack = foolbox.attacks.FGSM(model=foolmodel, criterion=TargetClass(target_class))
    elif attack_name == "cw":
        attack = foolbox.attacks.CarliniWagnerL2Attack(model=foolmodel, )
    elif attack_name == "jsma":
        attack = foolbox.attacks.SaliencyMapAttack(model=foolmodel, )
    elif attack_name == "fgsm":
        attack = foolbox.attacks.FGSM(model=foolmodel, )
    elif attack_name == "momentum":
        attack = foolbox.attacks.MomentumIterativeAttack(model=foolmodel, )
    elif attack_name == "noise_gauss":
        attack = foolbox.attacks.AdditiveGaussianNoiseAttack(model=foolmodel, )
    elif attack_name == "noise_salt":
        attack = foolbox.attacks.SaltAndPepperNoiseAttack(model=foolmodel, )
    elif attack_name == "bim_L2":
        attack = foolbox.attacks.L2BasicIterativeAttack(model=foolmodel, )
    else:
        raise ValueError()
    return attack


# 加载数据集
def load_data(data_name, use_norm=True):
    if data_name == model_conf.mnist:
        dau = MnistDau()
    elif data_name == model_conf.fashion:
        dau = FashionDau()
    elif data_name == model_conf.svhn:
        dau = SvhnDau()
    elif data_name == model_conf.cifar10:
        dau = CifarDau()
    else:
        raise ValueError()
    return dau.load_data(use_norm=use_norm)


# 对抗样本路径
def get_adv_path(data_name, model_name, attack_name, ori_label):
    pair_name = model_conf.get_pair_name(data_name, model_name)
    base_path = "{}/{}/{}/".format("adv_image", pair_name, attack_name, )
    os.makedirs(base_path, exist_ok=True)
    s_adv = "{}/{}.npy".format(base_path, ori_label)
    s_idx = "{}/{}_idx.npy".format(base_path, ori_label)
    return s_adv, s_idx


def get_adv_temp_path(data_name, model_name, attack_name, ori_label):
    pair_name = model_conf.get_pair_name(data_name, model_name)
    base_path = "{}/{}/{}/".format("adv_image_temp", pair_name, attack_name, )
    os.makedirs(base_path, exist_ok=True)
    s_adv = "{}/{}.npy".format(base_path, ori_label)
    s_idx = "{}/{}_idx.npy".format(base_path, ori_label)
    return s_adv, s_idx


def batch_gen_adv(data_name, model_name, attack_name):
    model_path = model_conf.get_model_path(data_name, model_name)
    (_, _), (x_test, y_test) = load_data(data_name)
    del _
    # futures = []
    sx_adv, sy_adv, s_idx = model_conf.get_adv_path_all(data_name, model_name, attack_name)  # 结果储存路径
    attack = get_attack(model_path, attack_name)  # prob
    lenth = int(len(x_test) * 0.5)
    x_test_1 = x_test[:lenth]
    y_test_1 = y_test[:lenth]
    x_test_2 = x_test[lenth:]
    y_test_2 = y_test[lenth:]

    adv_img_arr = []
    adv_lb_arr = []
    idx_arr = []
    num = 0
    for x_temp, y_temp in tqdm([(x_test_1, y_test_1), (x_test_2, y_test_2)]):
        adv_img = attack(x_temp, y_temp)
        assert len(adv_img) == len(x_temp)
        for j, (adv, lb) in enumerate(zip(adv_img, y_temp)):
            if np.isnan(adv).any():
                num += 1
            else:
                adv_img_arr.append(adv)
                adv_lb_arr.append(lb)
                idx_arr.append(j)
    print("adv_num:", len(adv_img_arr), "Nan_num:", num, "total_num:", len(x_test))
    np.save(sx_adv, np.array(adv_img_arr))
    np.save(sy_adv, np.array(adv_lb_arr))
    np.save(s_idx, np.array(idx_arr))


def exec(attack_name, data_name, model_name):
    print(attack_name, data_name, model_name)
    s = time.time()
    batch_gen_adv(data_name, model_name, attack_name)
    e = time.time()
    print("total time : {} m".format((e - s) / 60))
    K.clear_session()


# def test_adv(data_name, model_name, attack_name):
#     model_path = model_conf.get_model_path(data_name, model_name)
#     model = load_model(model_path)
#     # for ori_class in range(10):
#     #     for target_class in range(10):
#     #         if target_class == ori_class:
#     #             continue
#     #         s_adv = get_adv_path(data_name, model_name, attack_name, ori_class, target_class)[0]
#     #         adv_img = np.load(s_adv)
#     #         print("{} -> {}".format(ori_class, target_class))
#     #         print("adv_len : {}".format(len(adv_img)))
#     #         prob_matrixc = model.predict(adv_img)
#     #         y_max_prob_arr = np.max(prob_matrixc, axis=1)  #
#     #         plot_hist(y_max_prob_arr, ori_class, target_class, "target")
#     #         y_max_prob_arr = prob_matrixc[:, ori_class]
#     #         plot_hist(y_max_prob_arr, ori_class, target_class, "ori")
#
#     import matplotlib.pyplot as plt
#     plt.switch_backend('agg')
#     for ori_class in range(10):
#         for target_class in range(10):
#             if target_class == ori_class:
#                 continue
#             s_adv = get_adv_path(data_name, model_name, attack_name, ori_class, target_class)[0]
#             adv_img = np.load(s_adv)
#             print("{} -> {}".format(ori_class, target_class))
#             print("adv_len : {}".format(len(adv_img)))
#             prob_matrixc = model.predict(adv_img)
#             y_max_prob_arr = np.max(prob_matrixc, axis=1)  #
#             bins = np.linspace(min(y_max_prob_arr), max(y_max_prob_arr), 10)
#             # 这个是调用画直方图的函数，意思是把数据按照从bins的分割来画
#             plt.hist(y_max_prob_arr, bins)
#     # 设置出横坐标
#     plt.xlabel('Number of ×××')
#     # 设置纵坐标的标题
#     plt.ylabel('Number of occurences')
#     # 设置整个图片的标题
#     plt.title('Frequency distribution of number of ×××')
#     fig_dir = "hist"
#     os.makedirs(fig_dir, exist_ok=True)
#     plt.savefig("{}/adv.png".format(fig_dir))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    attack_list = ["ead", "pgd", "bim", "deepfool", "jsma", "fgsm"]

    for attack_name in attack_list:
        for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
            for model_name in model_name_arr:
                exec(attack_name, data_name, model_name)

    # ####### example
    # data_name = model_conf.mnist
    # model_name = model_conf.LeNet1
    # exec("fgsm", data_name, model_name)
