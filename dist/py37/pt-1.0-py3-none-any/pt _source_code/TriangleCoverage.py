import abc
import pt.BoundaryTriangle as BoundaryTriangle
from pt.util import get_data_by_label
import numpy as np

class TriangleCoverage(object):

    def __init__(self, bound_tri: BoundaryTriangle, is_save_profile=True, base_path=None, suffix=''):
        self.bound_tri = bound_tri
        self.is_save_profile = is_save_profile
        self.base_path = base_path
        self.suffix = suffix

    @staticmethod
    def get_p_q_list(n, i):
        num_list = list(range(n))
        num_list.remove(i)
        import itertools
        pq_list = []
        for pq in itertools.combinations(num_list, 2):
            pq_list.append(pq)

        return pq_list

    @staticmethod
    def get_total_bins_num(deep_num, dimensions_num):
        return 4 ** deep_num * dimensions_num

    @abc.abstractmethod
    def cal_triangle_cov(self, Tx, Ty, n, deep_num, by_deep_num=True):
        pass


class BaseTriCoverage(TriangleCoverage):

    def cal_triangle_cov(self, Tx, Ty, n, deep_num, by_deep_num=True):
        if by_deep_num:
            cov, var, num = self.cal_cov_fine(Tx, Ty, n, deep_num)
        else:
            cov, var, num = self.cal_cov_coarse(Tx, Ty, n, deep_num)
        return (
         cov, var)

    def cal_triangle_cov_with_num(self, Tx, Ty, n, deep_num, by_deep_num=True):
        if by_deep_num:
            cov, var, num = self.cal_cov_fine(Tx, Ty, n, deep_num)
        else:
            cov, var, num = self.cal_cov_coarse(Tx, Ty, n, deep_num)
        return (
         cov, var, num)

    def cal_cov_fine(self, Tx_prob_matrix, Ty, n, deep_num):
        cov_rate_arr_list = []
        cov_num_arr_list = []
        for i in range(n):
            csv_data = {}
            Tx_prob_matrixc_i, Ty_i = get_data_by_label(Tx_prob_matrix, Ty, i)
            if Tx_prob_matrixc_i.size == 0:
                cov_rate_arr_list.append([0] * deep_num)
                cov_num_arr_list.append([0] * deep_num)
                continue
            pq_list = self.get_p_q_list(n, i)
            pq_cov_len_arr_list = []
            for p, q in pq_list:
                bins_arr = []
                for deep_num_ix in range(deep_num):
                    bins = [
                     0] * 4 ** (deep_num_ix + 1)
                    bins_arr.append(bins)

                S0_projection_matrixc = self.bound_tri.get_projection_matrixc(Tx_prob_matrixc_i, p, q, n, i)
                for x_k in S0_projection_matrixc:
                    x_ = x_k[i]
                    y_ = x_k[p]
                    z_ = x_k[q]
                    k_list = self.bound_tri.get_position(x_, y_, deep_num)
                    for deep_num_ix in range(deep_num):
                        k = k_list[deep_num_ix]
                        bins_arr[deep_num_ix][k] = 1

                pq_cov_len_arr = []
                for bins in bins_arr:
                    pq_cov_len = np.sum(bins)
                    pq_cov_len_arr.append(pq_cov_len)

                pq_cov_len_arr_list.append(pq_cov_len_arr)

            total_cov_len_list = np.sum(pq_cov_len_arr_list, axis=0)
            csv_data['label'] = i
            cov_rate_arr = []
            cov_num_arr = []
            for deep_num_ix in range(deep_num):
                total_cov_len = total_cov_len_list[deep_num_ix]
                cov_num_arr.append(total_cov_len)
                cov_rate = total_cov_len / (4 ** (deep_num_ix + 1) * len(pq_list))
                csv_data['cov_rate_{}'.format(deep_num_ix)] = cov_rate
                cov_rate_arr.append(cov_rate)

            cov_rate_arr_list.append(cov_rate_arr)
            cov_num_arr_list.append(cov_num_arr)

        if len(cov_rate_arr_list) != n:
            print(len(cov_rate_arr_list))
            raise ValueError('len c_arr  not eq nbclasses')
        cov_rate_arr = np.mean(cov_rate_arr_list, axis=0)
        cov_num_arr = np.sum(cov_num_arr_list, axis=0)
        cov_var_arr = np.var(cov_rate_arr_list, axis=0)
        return (cov_rate_arr, cov_var_arr, cov_num_arr)

    def cal_cov_coarse(self, Tx_prob_matrix, Ty, n, deep_num, base_path=None, suffix=''):
        cov_rate_arr = []
        cov_num_arr = []
        for i in range(n):
            csv_data = {}
            Tx_prob_matrix_i, Ty_i = get_data_by_label(Tx_prob_matrix, Ty, i)
            if Tx_prob_matrix_i.size == 0:
                cov_rate_arr.append(0)
                cov_num_arr.append(0)
                continue
            pq_list = self.get_p_q_list(n, i)
            pq_cov_len_arr = []
            for p, q in pq_list:
                bins = [
                 0] * 4 ** deep_num
                S0_projection_matrixc = self.bound_tri.get_projection_matrixc(Tx_prob_matrix_i, p, q, n, i)
                for x_k in S0_projection_matrixc:
                    x_ = x_k[i]
                    y_ = x_k[p]
                    z_ = x_k[q]
                    k_list = self.bound_tri.get_position(x_, y_, deep_num)
                    k = k_list[(-1)]
                    bins[k] = 1

                pq_cov_len = np.sum(bins)
                pq_cov_len_arr.append(pq_cov_len)

            total_cov_len = np.sum(pq_cov_len_arr)
            cov_rate = total_cov_len / self.get_total_bins_num(deep_num, len(pq_list))
            cov_rate_arr.append(cov_rate)
            cov_num_arr.append(total_cov_len)
            csv_data['label'] = i
            csv_data['cov_rate'] = cov_rate

        if len(cov_rate_arr) != n:
            print(len(cov_rate_arr))
            raise ValueError('len c_arr  not eq nbclasses')
        cov_rate = np.mean(cov_rate_arr)
        cov_var = np.var(cov_rate_arr)
        cov_num = np.sum(cov_num_arr)
        return (cov_rate, cov_var, cov_num)
