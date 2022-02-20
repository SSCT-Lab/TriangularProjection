import abc

from utils import model_conf


class MyFig(object):
    def __init__(self, tab):
        self.output_fig_dir_path = "figs"
        self.tab = tab
        self.fig_suffixes = ["pdf", "png"]
        self.init_dirs()

    @abc.abstractmethod
    def plot_figs(self):
        ...

    @abc.abstractmethod
    def init_dirs(self):
        ...

    @staticmethod
    def get_color_dict():
        dict = {
            "sp_c_4": "C3",
            "DeepSpace": "C3",
            "Pt": "crimson",
            "random": "black",
            "Random": "black",
            "cov_lsc": "C0",
            "LSC": "C0",
            "cov_nac": "C1",
            "NAC": "C1",
            "cov_nbc": "C2",
            "NBC": "C2",
            "cov_snac": 'C4',
            "SNAC": 'C4',
            "cov_tknc": "C5",
            "TKNC": "C5",
            "cov_kmnc": "C6",
            "KMNC": "C6",
        }
        return dict

    def get_lb_dict(self):
        dict1 = self.tab.get_df2tab_index_dict()
        params = {
            "NAC": "NAC",
            "NBC": "NBC",
            "SNAC": "SNAC",
            "TKNC": "TKNC",
            "KMNC": "KMNC",
            "LSC": "LSC",
            "DeepSpace": "Pt",
            "Random": "Random",
            "random": "Random",
        }
        dict2 = dict(dict1, **params)
        return dict2

    def get_figs_name(self, data_name, model_name):
        return model_conf.get_pair_name(data_name, model_name)
