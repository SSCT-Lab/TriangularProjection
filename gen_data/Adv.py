# 已经是浮点数了
from utils import model_conf
import numpy as np


class MyAdv(object):
    def __init__(self):
        self.cache_map = {}
        pass

    # def load_adv_data(self, attack_name, data_name, model_name):
    #     x_adv_arr = []
    #     y_adv_arr = []
    #     for i in range(10):
    #         im = model_conf.get_adv_path(attack_name, data_name, model_name, i)
    #         x_adv = np.load(im)
    #         y_adv = np.array([i] * len(x_adv))
    #         x_adv_arr.append(x_adv)
    #         y_adv_arr.append(y_adv)
    #     x_adv_arr = np.concatenate(x_adv_arr, axis=0)
    #     y_adv_arr = np.concatenate(y_adv_arr, axis=0)
    #     # print("adv", len(x_adv_arr), len(y_adv_arr))
    #     return x_adv_arr, y_adv_arr

    # get_adv_path_all

    def load_adv_data(self, attack_name, data_name, model_name, use_cache=False):
        if use_cache:
            key = attack_name + "_" + data_name + "_" + model_name
            sx_adv, sy_adv, s_idx = model_conf.get_adv_path_all(data_name, model_name, attack_name)
            x_adv_arr = np.load(sx_adv)
            y_adv_arr = np.load(sy_adv)
            if key not in self.cache_map.keys():
                self.cache_map[key + "_x"] = x_adv_arr
                self.cache_map[key + "_y"] = y_adv_arr
            else:
                return self.cache_map[key + "_x"], self.cache_map[key + "_y"]
        else:
            # 不使用缓存
            sx_adv, sy_adv, s_idx = model_conf.get_adv_path_all(data_name, model_name, attack_name)
            x_adv_arr = np.load(sx_adv)
            y_adv_arr = np.load(sy_adv)
        return x_adv_arr, y_adv_arr

    def clear_cache(self):
        print("clear key:{}".format(self.cache_map.keys()))
        self.cache_map = {}