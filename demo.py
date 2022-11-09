import os
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from termcolor import colored
from gen_data.MnistDau import MnistDau
from pt import TriProCover
from utils import model_conf
from utils.utils import num_to_str, shuffle_data


def color_print(s, c):
    print(colored(s, c))


class DemoMnistDau(MnistDau):
    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
        }
        return params


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=10, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


# a demo for pt
if __name__ == '__main__':
    # initial TriangularProjectionCover
    base_path = "demo"
    os.makedirs(base_path, exist_ok=True)
    deep_num = 4
    tripro_cover = TriProCover()

    # mnist data
    color_print("load LeNet-5 model and MNIST data sets", "blue")
    dau = DemoMnistDau()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    test_size = dau.test_size
    nb_classes = model_conf.fig_nb_classes

    # LeNet5 model
    model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
    ori_model = load_model(model_path)

    acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10))[1]
    print("accuracy {}".format(acc))

    # metrics
    color_print("calculate ori test pt coverage", "blue")
    x_test_prob_matrix = ori_model.predict(x_test)
    sp_c_arr, sp_v_arr = tripro_cover.cal_triangle_cov(x_test_prob_matrix, y_test, nb_classes, deep_num,
                                                       by_deep_num=True)

    sp_c_str_arr = [num_to_str(x, 5) for x in sp_c_arr]
    for i, ratio in enumerate(sp_c_str_arr):
        print("ori data. deep: {} pt coverage: {}".format(i, ratio))

    # # data augmentation
    color_print("data augmentation", "blue")
    dau.run("test")
    x_dau, y_dau = dau.load_dau_data("SF", use_cache=False)

    # metrics
    color_print("calculate aug test pt coverage", "blue")
    x_dau_prob_matrix = ori_model.predict(x_dau)
    sp_c_arr, sp_v_arr = tripro_cover.cal_triangle_cov(x_dau_prob_matrix, y_dau, nb_classes, deep_num,
                                                       by_deep_num=True)

    sp_c_str_arr = [num_to_str(x, 5) for x in sp_c_arr]
    for i, ratio in enumerate(sp_c_str_arr):
        print("shift data. deep: {} pt coverage: {}".format(i, ratio))

    # selection
    color_print("test selection on aug test", "blue")
    x_dau, y_dau = shuffle_data(x_dau, y_dau)
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10))[1]
    x_sel_prob_matrix = ori_model.predict(x_sel)
    xs, ys, ix_arr, cov_rate, max_cov_num = tripro_cover.rank_greedy(x_sel, x_sel_prob_matrix, y_sel, nb_classes,
                                                                     deep_num)
    xs = np.array(xs)
    print("max_cov_num: {} total_test_size: {}".format(max_cov_num, len(x_sel)))

    # retrain
    # pt
    num = 1000
    color_print("pt. retrain model with {} selected aug data".format(num), "blue")
    filepath = os.path.join(base_path, "pt_temp.h5")
    trained_model = train_model(ori_model, filepath, xs[:num],
                                keras.utils.np_utils.to_categorical(ys[:num], 10), x_val,
                                keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)
    acc_val1 = trained_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10))[1]
    print("retrain model path: {}".format(filepath))
    print("pt. train acc improve {} -> {}".format(acc_val0, acc_val1))

    # random
    color_print("random. retrain model with {} selected aug data".format(num), "blue")
    ori_model = load_model(model_path)
    filepath = os.path.join(base_path, "rn_temp.h5")
    xr, yr = shuffle_data(x_sel, y_sel)
    trained_model = train_model(ori_model, filepath, xr[:num],
                                keras.utils.np_utils.to_categorical(yr[:num], 10), x_val,
                                keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)
    acc_val2 = trained_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10))[1]
    print("retrain model path: {}".format(filepath))
    print("random. train acc improve {} -> {}".format(acc_val0, acc_val2))

    color_print("Accuracy difference between pt and random :{}".format(acc_val1 - acc_val2), "green")
