import numpy as np
import pt.TriangleCoverage as TriangleCoverage

class TriCovFuzz(TriangleCoverage):

    def cal_cov_bins(self, Tx, Ty, n, deep_num):
        lb_arr = np.unique(Ty)
        i = lb_arr[0]
        assert len(set(lb_arr)) == 1
        Tx_prob_matrixc, Ty_i = Tx, Ty
        if Ty.size == 0:
            raise ValueError('lable i do not have data')
        pq_list = self.get_p_q_list(n, i)
        res_arr = []
        for x in Tx_prob_matrixc:
            bins_pq = []
            for p, q in pq_list:
                bins = [
                 0] * 4 ** deep_num
                x_k = self.bound_tri.get_projection_point(i, p, q, n, x)
                x_ = x_k[i]
                y_ = x_k[p]
                z_ = x_k[q]
                k_list = self.bound_tri.get_position(x_, y_, deep_num)
                k = k_list[(-1)]
                bins[k] = 1
                bins_pq.append(bins)

            bins_pq = np.concatenate(bins_pq, axis=0)
            res_arr.append(bins_pq)

        return np.array(res_arr)
