import numpy as np

class BoundaryTriangle(object):

    @staticmethod
    def get_projection_matrixc(X, p, q, n, i):
        x_k_dot_matrixc = []
        for x_k in X:
            x_k_dot = BoundaryTriangle.get_projection_point(i, p, q, n, x_k)
            x_k_dot_matrixc.append(x_k_dot)

        return np.array(x_k_dot_matrixc)

    @staticmethod
    def get_projection_point(i, p, q, n, A):
        one_third = 0.3333333333333333
        two_third = 0.6666666666666666
        A_dot = np.zeros(A.shape)
        A_dot[i] = two_third * A[i] - one_third * A[p] - one_third * A[q] + one_third
        A_dot[p] = two_third * A[p] - one_third * A[q] - one_third * A[i] + one_third
        A_dot[p] = two_third * A[q] - one_third * A[p] - one_third * A[i] + one_third
        return A_dot

    @staticmethod
    def get_position(x, y, deep_num):
        triangle_first = [[0, 0], [1, 0], [0, 1]]
        is_reversed = False
        k = 0
        k_list = []
        for i in range(deep_num):
            triangle_first, k, is_reversed = BoundaryTriangle.get_position_detail(x, y, triangle_first, k, is_reversed)
            k_list.append(k)

        return k_list

    @staticmethod
    def get_position_detail(x, y, triangle_pre, k, is_reversed):
        (p1x, p1y), (p2x, p2y), (p3x, p3y) = triangle_pre
        midpoint_x = (p1x + p2x) / 2
        midpoint_y = (p1y + p3y) / 2
        a = midpoint_x
        b = midpoint_y
        if not is_reversed:
            if x >= a:
                k = k * 4 + 0
                triangle_cur = [[a, p1y], [p2x, p2y], [a, b]]
            else:
                if y >= b:
                    triangle_cur = [
                     [
                      p1x, a], [a, b], [p3x, p3y]]
                    k = k * 4 + 1
                else:
                    if x + y <= a + p1y:
                        k = k * 4 + 2
                        triangle_cur = [[p1x, p1y], [a, p2y], [p3x, b]]
                    else:
                        k = k * 4 + 3
                        triangle_cur = [[a, p1y], [p3x, b], [a, b]]
                        is_reversed = not is_reversed
        else:
            if x <= a:
                k = k * 4 + 0
                triangle_cur = [[a, b], [p2x, p2y], [a, p3y]]
            else:
                if y < b:
                    k = k * 4 + 1
                    triangle_cur = [[p1x, p1y], [a, b], [p3x, b]]
                else:
                    if x + y > a + p2y:
                        k = k * 4 + 2
                        triangle_cur = [[p1x, b], [a, p2y], [p3x, p3y]]
                    else:
                        k = k * 4 + 3
                        triangle_cur = [[a, b], [p1x, b], [a, p3y]]
                        is_reversed = not is_reversed
        return (
         triangle_cur, k, is_reversed)


