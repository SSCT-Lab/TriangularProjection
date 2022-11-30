from tqdm import tqdm
import numpy as np
import pt.TriangleCoverage as TriangleCoverage

class TriCovRank(TriangleCoverage):

    def rank_greedy(self, Tx, Tx_prob_matrix, Ty, n, deep_num, use_pretreatment=None):
        return self.guide_selection(Tx, Tx_prob_matrix, Ty, n, deep_num)

    def guide_selection(self, Tx, Tx_prob_matrix, Ty, n, deep_num, use_add=False):
        x_guide = []
        y_guide = []
        ix_guide = []
        len_cov_arr = [
         0] * n
        bins_pq_dict = {}
        len_pq_list = 0
        for i in range(n):
            pq_list = self.get_p_q_list(n, i)
            len_pq_list = len(pq_list)
            bins_pq = {}
            for p, q in pq_list:
                bins = [
                 0] * 4 ** deep_num
                bins_pq[(p, q)] = bins

            bins_pq_dict[i] = bins_pq

        x_others = []
        y_others = []
        ix_others = []
        for ix, (x, x_prob, y) in tqdm(enumerate(zip(Tx, Tx_prob_matrix, Ty))):
            add_flag = False
            i = y
            bins_pq = bins_pq_dict[i]
            pq_list = self.get_p_q_list(n, i)
            add_cov_num = 0
            for p, q in pq_list:
                bins = bins_pq[(p, q)]
                x_k = self.bound_tri.get_projection_point(i, p, q, n, x_prob)
                x_ = x_k[i]
                y_ = x_k[p]
                z_ = x_k[q]
                k_list = self.bound_tri.get_position(x_, y_, deep_num)
                k = k_list[(-1)]
                if bins[k] == 0:
                    add_flag = True
                    bins[k] = 1
                    add_cov_num += 1

            len_cov_arr[i] += add_cov_num
            if add_flag:
                ix_guide.append(ix)
                x_guide.append(x)
                y_guide.append(y)
            else:
                ix_others.append(ix)
                x_others.append(x)
                y_others.append(y)

        total_bins = self.get_total_bins_num(deep_num, len_pq_list)
        rate_cov_arr = np.array(len_cov_arr) / total_bins
        cov_rate = np.mean(rate_cov_arr)
        max_cov_num = len(x_guide)
        if use_add:
            x_guide = np.concatenate([x_guide, x_others], axis=0)
            y_guide = np.concatenate([y_guide, y_others], axis=0)
            ix_guide = np.concatenate([ix_guide, ix_others], axis=0)
        return (
         x_guide, y_guide, ix_guide, cov_rate, max_cov_num)

    def rank_ctm(self, Tx, Ty, n, M, deep_num):
        Tx_prob_matrixc = M.predict(Tx)
        x_guide = []
        y_guide = []
        len_cov_arr = [
         0] * n
        bins_pq_dict = {}
        len_pq_list = 0
        for i in range(n):
            pq_list = self.get_p_q_list(n, i)
            len_pq_list = len(pq_list)
            bins_pq = {}
            for p, q in pq_list:
                bins = [
                 0] * 4 ** deep_num
                bins_pq[(p, q)] = bins

            bins_pq_dict[i] = bins_pq

        cov_num_arr = []
        for ix, (x, x_prob, y) in tqdm(enumerate(zip(Tx, Tx_prob_matrixc, Ty))):
            i = y
            pq_list = self.get_p_q_list(n, i)
            add_cov_num = 0
            for p, q in pq_list:
                x_k = self.bound_tri.get_projection_point(i, p, q, n, x_prob)
                x_ = x_k[i]
                y_ = x_k[p]
                z_ = x_k[q]
                k_list = self.bound_tri.get_position(x_, y_, deep_num)
                k = k_list[(-1)]
                if k != 0:
                    add_cov_num += 1

            cov_num_arr.append(add_cov_num)

        assert len(cov_num_arr) == len(Tx)
        return np.argsort(cov_num_arr)[::-1]
