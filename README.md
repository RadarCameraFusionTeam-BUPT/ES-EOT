# ES-EOT

If you use the code, please cite our paper:

```text
@article{deng20243d,
  title={3D Extended Object Tracking by Fusing Roadside Sparse Radar Point Clouds and Pixel Keypoints},
  author={Jiayin Deng and Zhiqun Hu and Yuxuan Xia and Zhaoming Lu and Xiangming Wen},
  journal={arXiv preprint arXiv:2404.17903},
  year={2024},
  doi={2404.17903}
}
```

## Installation

* Clone the repository and cd to it.

    ```bash
    git clone https://github.com/RadarCameraFusionTeam-BUPT/ES-EOT.git
    cd ES-EOT
    ```

* Create and activate virtual environment using anaconda (tested on python 3.8 and 3.10).

    ```bash
    conda create -n ES_EOT python=3.8
    conda activate ES_EOT
    ```

* Install dependencies for yolov8.

    ```bash
    conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
    ```

* Install other dependencies.

    ```bash
    pip install scikit-learn
    pip install shapely
    ```

## Prepare simulation data

* Download simulation data from [data link](https://drive.google.com/file/d/1vYbKewCbCrHDxhkEiPgm3cb5P3yPoCR9/view?usp=drive_link), and unzip into the `data` folder.

## Usage (`ES-EOT` as an example)

**Note**: Different EOT methods are located in different folders, and each method can be executed in two ways.

### 1. Test on one of the data

* Run the main file. (The result is written in a .npy file in the same folder as main.py)

    ```bash
    cd ES-EOT
    python main.py
    ```

* Calculate the IOU values and RMSE of velocity.

    ```bash
    python calculate_matrics.py
    ```

* Show the tracking results in an animation.

    ```bash
    python ShowRes.py
    ```

### 2. Test on dataset

* Run the batch_main file. (The results are written in the `res` folder)

    ```bash
    cd ES-EOT
    python batch_main.py
    ```

* Calculate the IOU values and RMSE of velocities.

    ```bash
    python batch_calculate_matrics.py
    ```

## Show

* Show IOU with frame in a .jpg picture

    ```bash
    cd show
    python batch_calculate_matrics.py
    ```

* Show IOU and velocity RMSE in a table

    ```bash
    cd show
    python iou_rmsev_table.py
    ```

## Keypoints detection using pre-trained model

Note: Keypoints detections are stored in the `data/<scene>/vision/output-keypoints.npy`, if you've prepared the dataset as described above. However, if you wish to reuse the pre-trained model, follow these steps:

### 1. Download the pre-trained model

* Obtain the pre-trained model weights from [model link](https://drive.google.com/file/d/1vnJbfMzvKxIPGX49Lkmc9Tlr9XrGiv-I/view?usp=drive_link), and move it into the `assets` folder.

### 2. Run Keypoints Detection

* Once you have the pre-trained model weights and dependencies installed, you can run the key points detection script

    ```bash
    python keypoints_det.py data/bus_change_lane/vision/output.mp4 --model assets/best.pt --render
    ```

    The results are written into the `output` folder.
