This folder contains the raw estimated results and the evaluation script used for our paper. Change working directory to `./paper` and run `python3 evaluate.py`. Make sure to change the base directories of the datasets at the bottom of `evaluate.py`.

`kitti_metric.py` contains the code for computing the KITTI-error metric. This is a reimplementation of the original error metric wirtten in C++. Visit [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) for the original implementation.

`estimates` contains the results for KISS-ICP, KISS-SLAM, MAD-ICP and our method. The source code of MAD-ICP was adapted to write the needed registration times into a separate file.


