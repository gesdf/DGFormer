# DGFormer: Depth-Geometry Fusion Transformer for Spatial Relationship Recognition 

<img width="1279" height="562" alt="image" src="https://github.com/user-attachments/assets/aa76393f-caf8-44fe-a4e3-5914ceac2583" />

# Getting Started

This code is based on the [spatial-relation-benchmark](https://github.com/AlvinWen428/spatial-relation-benchmark).

First clone our repository by 

```
git clone git@github.com:gesdf/DGFormer.git
```

## Installation

The environment installation can follow the settings of the [spatial-relation-benchmark](https://github.com/AlvinWen428/spatial-relation-benchmark).
```
pip install -r requirements.txt
```
## Download SpatialSense, SpatialSense, and the IBOT-pretrained ViT backbone.

Make sure you are in `DGFormer`, download.py can be used for downloading the SpatialSense,SpatialSense+, and the IBOT pretrained backbone:


# Experiments

## Training

`main.py` is the entry of all experiments. All the experiment configs for training and evaluating all the models on SpatialSense and SpatialSense+ can be found in `configs/`.
The training process can be executed by:

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10000 --nproc_per_node 4 python main.py --exp-config ${CONFIG_PATH} [Args1, Args2, ...]
```

## Testing

If you have trained your model and obtained the weights, you can conduct the test in the following way:

```
python main.py --entry batch-test --model-path results/spatialsense_DGFormer
```


