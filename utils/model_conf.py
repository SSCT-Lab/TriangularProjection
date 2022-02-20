import os

image = "image"
label = "label"
mnist = "mnist"
fashion = "fashion"
cifar10 = "cifar"
svhn = "svhn"

LeNet5 = "LeNet5"
LeNet1 = "LeNet1"
resNet20 = "resNet20"
vgg16 = "vgg16"
MyLeNet5 = "MyLeNet5"
MyVgg16 = "MyVgg16"

fig_nb_classes = 10

name_list = [mnist, fashion, svhn, cifar10]
model_data = {
    mnist: [LeNet5, LeNet1],
    fashion: [LeNet1, resNet20],
    cifar10: [vgg16, resNet20],
    svhn: [LeNet5, vgg16]
}

pair_list = ["mnist_LeNet5", "mnist_LeNet1", "fashion_resNet20", "fashion_LeNet1", "svhn_LeNet5", "svhn_vgg16",
             "cifar_resNet20", "cifar_vgg16"]


# 模型位置
def get_model_path(datasets, model_name):
    dic = {"mnist_LeNet5": './model/model_mnist_LeNet5.hdf5',
           "mnist_LeNet1": "./model/model_mnist_LeNet1.hdf5",
           "fashion_resNet20": "./model/model_fashion_resNet20.hdf5",
           "fashion_LeNet1": "./model/model_fashion_LeNet1.hdf5",
           "cifar_vgg16": "./model/model_cifar_vgg16.hdf5",
           "cifar_resNet20": "./model/model_cifar_resNet20.h5",
           "svhn_vgg16": "./model/model_svhn_vgg16.hdf5",
           "svhn_LeNet5": "./model/model_svhn_LeNet5.hdf5",
           "cifar100_LeNet5": './model/model_cifar100_LeNet5.hdf5',
           "cifar100_vgg16": './model/model_cifar100_vgg16.hdf5',
           }
    return dic[datasets + "_" + model_name]


def get_temp_model_path(datasets, model_name, smaple_method):
    path = './temp_model/' + datasets + "/" + model_name + "/" + smaple_method
    return path


def get_adv_path(attack, dataset, model_name, ori_label):
    dir_path = "adv_image"
    i = '{}/{}_{}/{}/{}.npy'.format(dir_path, dataset, model_name, attack, ori_label)
    return i


def get_adv_path_all(data_name, model_name, attack_name):
    pair_name = get_pair_name(data_name, model_name)
    base_path = "{}/{}".format("adv_image", pair_name)
    os.makedirs(base_path, exist_ok=True)
    sx_adv = "{}/{}_{}.npy".format(base_path, attack_name, "x")
    sy_adv = "{}/{}_{}.npy".format(base_path, attack_name, "y")
    s_idx = "{}/{}_idx.npy".format(base_path, attack_name)
    return sx_adv, sy_adv, s_idx


def get_pair_name(data_name, model_name):
    return data_name + "_" + model_name
