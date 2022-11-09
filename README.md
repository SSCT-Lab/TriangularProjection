# Triangular Projection

This repository contains a  brand new diversity-based testing criterion , `triangular projection(pt)` , for testing deep neural networks.

![](https://cdn.mathpix.com/snip/images/hQQFvFCoS3u90fO9xdCGukp_OwXzb3jxafs0_qjU2p8.original.fullsize.png)

## Introduction

Different from the traditional idea of structure coverage, we do not start from the activation state of neurons, but from the perspective of **the diversity of the DNNs' output layer**, we obtain a test suite evaluation criterion of the current model.

Based on the application scenarios of detection, selection and generation, we fully verified the effectiveness of triangular projection in DNNs testing.

## Installation

```
pip install -r requirements.txt
pip install ./dist/py37/pt-1.0-py3-none-any.whl
```

## The structure of the repository

In the experiment, our method and all baselines are conducted upon `python 3.7` with` Keras 2.3.1` and  `TensorFlow 1.13.1`. All experiments are performed on a `Ubuntu 18.04.3 LTS server` with `two NVIDIA Tesla V100 GPU`, one 10-core processor "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz", and `120GB memory`.

main_folder:

```
├── gen_data/ "load data"
├── gen_model/ "to get the model in experiment" 
├── gen_table/ "to get the pictures and tables in experiment"
├── nc_coverage/ "neuron coverage test criteria"
├── tri_projection/ "triangular projection test criteria"
├── utils/ "some tool functions"
├── exp_correlation.py  "RQ1"
├── exp_selection.py	"RQ2"
├── gen_adv.py "to generate adversarial examples"
├── statistics "a interface to get the pictures and tables in experiment"
├── init.py "init dirs and download svhn data"
├── demo.py "a demo for pt"
```

others:

```
├── result/ "the raw output of experiment"
├── figs/ "pictures of experimental results"
├── tabs/ "tables of experimental results"
├── data/ "svhn data"
├── adv_image/ "adversarial images"
├── dau/ "Data Augmentation"
├── model/ "keras model"
├── temp_model/ "temp files"
├── README.md
└── requirements.txt
```

## Usage

We prepared a demo for Pt

- `python demo.py`.

If you want to reproduce our experiment:

1. initial  models and datasets

   - we can download by this link

   - link：https://pan.baidu.com/s/1O_Pwa5Q6feOixxS0UNXFLg 

     code：trpr

   - or initial  by python files

     1. initial dirs , svhn data and models

        `python init.py`

     2. data augmentation

        `python -m gen_data.{MnistDau}/{CifarDau}/{FashionDau}/{SvhnDau}`

     3. adversarial images

        `python gen_adv.py`

3. experiment

   - RQ1

     `python exp_correlation.py`

   - RQ2

     `python exp_selection.py`

   - RQ3

     `cd deephunter ` and `python run_fuzzer.py`

     see `deephunter/readme.md`
     
     Before run this experiment,you should prepare models and datasets (see `Customization` in `readme.md`) ,or download by link.

4. get results

   `python statistics.py`

## Experimental result

1. Measuring the diversity of test suites

   ![](https://cdn.mathpix.com/snip/images/6VRLD-4u9SJp75SRO3eMT4d3QRiMcB-kmykRXvrTjE8.original.fullsize.png)

2. Guiding the model optimization

   ![](https://cdn.mathpix.com/snip/images/cs1wLPFK2TziDYQf3bE2mRxGvGPtmoXcZfV8X4zYfZ0.original.fullsize.png)

3. Guiding the test generation

![](https://cdn.mathpix.com/snip/images/Jq3CElY5jRUEPIz7dIE6-wgiroGtgJwoKR3qsmaQyhs.original.fullsize.png)

## Citation

Please cite the following paper if `Triangular Projection` helps you on the research:

```.

```

