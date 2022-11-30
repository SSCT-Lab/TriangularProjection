from typing import Tuple

import numpy as np


class TriProCover:

    def __init__(self) -> None: ...

    '''
    :param Tx_prob_matrix:  m * n matrix
    :param Ty: labels
    :param n: Number of classifications or Number of dimensions
    :param deep_num: Triangle division depth
    :param by_deep_num: If it is false, only the coverage of the last layer is calculated. If it is true, the coverage of each layer is output
    :return: Total coverage rate and variance of each dimension
    '''

    def cal_triangle_cov(self, Tx_prob_matrix: np.ndarray, Ty: np.ndarray, n: int, deep_num: int,
                         by_deep_num: bool = ...) -> Tuple[float, float]: ...

    '''
    :param Tx: ori data
    :param Tx_prob_matrix:  m * n matrix
    :param Ty: labels
    :param n: Number of classifications or Number of dimensions
    :param deep_num: Triangle division depth
    :return: cov-guided selected data x, cov-guided selected data y, index of data, coverage rate, Maximum coverage num
    '''

    def rank_greedy(self, Tx: np.ndarray, Tx_prob_matrix: np.ndarray, Ty: np.ndarray, n: int, deep_num: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float, int]: ...
