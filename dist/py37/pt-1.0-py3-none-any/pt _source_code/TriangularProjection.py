import pt.BoundaryTriangle as BoundaryTriangle
import pt.TriCovFuzz as TriCovFuzz
import pt.TriCovRank as TriCovRank
from pt.TriangleCoverage import BaseTriCoverage
import numpy as np

class TriProCover(object):

    def __init__(self, is_save_profile=False, base_path: str=None, suffix=''):
        bound_tri = BoundaryTriangle()
        self.bound_tri = bound_tri
        tri_cov = BaseTriCoverage(bound_tri, is_save_profile=is_save_profile, base_path=base_path, suffix=suffix)
        self.tri_cov = tri_cov
        tri_rank = TriCovRank(bound_tri, is_save_profile=is_save_profile, base_path=base_path, suffix=suffix)
        self.tri_rank = tri_rank
        tri_ptr = TriCovFuzz(bound_tri, is_save_profile=is_save_profile, base_path=base_path, suffix=suffix)
        self.tri_ptr = tri_ptr

    def cal_triangle_cov(self, Tx_prob_matrix: np.ndarray, Ty: np.ndarray, n: int, deep_num: int, by_deep_num=True) -> (
 float, float):
        return self.tri_cov.cal_triangle_cov(Tx_prob_matrix, Ty, n, deep_num, by_deep_num=by_deep_num)

    def rank_greedy(self, Tx: np.ndarray, Tx_prob_matrix: np.ndarray, Ty: np.ndarray, n: int, deep_num: int) -> (
 np.ndarray, np.ndarray, np.ndarray, float, int):
        return self.tri_rank.rank_greedy(Tx, Tx_prob_matrix, Ty, n, deep_num, use_pretreatment=None)

    def cal_cov_bins(self, Tx: np.ndarray, Ty: np.ndarray, n: int, deep_num: int):
        self.tri_ptr.cal_cov_bins(Tx, Ty, n, deep_num)

    def get_dimensions_num(self, n: int):
        return len(self.tri_cov.get_p_q_list(n, 0))

    def get_total_bins_num(self, deep_num: int, dimensions_num: int):
        return self.tri_cov.get_total_bins_num(deep_num, dimensions_num)
