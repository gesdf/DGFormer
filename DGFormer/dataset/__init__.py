import torch
from torch.utils.data import DataLoader, DistributedSampler

from .rel3d import Rel3dDataset
from .spatial_sense import SpatialSenseDataset,get_grouped_loader
import utils
# 定义创建数据集和数据加载器的函数
def create_dataset(split, **kwargs):
    """创建指定分割的数据集"""
    return SpatialSenseDataset(split=split, **kwargs)

def create_dataloader(split, dataset_name, predicate_dim, object_dim, datapath, num_workers,
                      crop, norm_data, load_img, data_aug_shift,
                      data_aug_color, batch_size, resize_mask, trans_vec,category_map_path,train_valid=False, shuffle=None):
    if dataset_name == "rel3d":
        dataset_args = {
            "split": split,
            "predicate_dim": predicate_dim,
            "object_dim": object_dim,
            "data_path": datapath,
            "load_img": load_img,
            "data_aug_shift": (data_aug_shift and split == "train"),
            "data_aug_color": (data_aug_color and split == "train"),
            "crop": crop,
            "norm_data": norm_data,
            "resize_mask": resize_mask,
            "trans_vec": trans_vec,
            "train_valid": train_valid
        }
        dataset = Rel3dDataset(**dataset_args)
    elif dataset_name == "spatialsense":
        dataset_args = {
            "split": split,
            "predicate_dim": predicate_dim,
            "object_dim": object_dim,
            "data_path": datapath,
            "load_img": load_img,
            "data_aug_shift": (data_aug_shift and split == "train"),
            "data_aug_color": (data_aug_color and split == "train"),
            "crop": crop,
            "norm_data": norm_data,
            "category_map_path": category_map_path,
            "train_valid": train_valid
        }
        dataset = SpatialSenseDataset(**dataset_args)
    else:
        raise ValueError
    if utils.is_dist_avail_and_initialized():
        world_size = utils.get_world_size()
        assert batch_size % world_size == 0, "the batch_size should be divisible by world_size"
        if split == "train":
            sampler = DistributedSampler(
                dataset, shuffle=(split == "train"),
                # drop_last=(split == "train"),
            )
            dataloader = DataLoader(dataset, batch_size=batch_size // world_size, sampler=sampler,
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size // world_size, num_workers=num_workers,
                                    shuffle=shuffle or (split == "train"),
                                    # drop_last=(split == "train"),
                                    pin_memory=torch.cuda.is_available())
    else:
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers,
                                shuffle=shuffle or (split == "train"),
                                # drop_last=(split == "train"),
                                pin_memory=torch.cuda.is_available())

    return dataloader
