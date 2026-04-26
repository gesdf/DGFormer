from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.MODEL_NAME = '2d'
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 0
_C.EXP.OUTPUT_DIR = "./results"
# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.dataset_name = "rel3d"
_C.DATALOADER.batch_size = 128
_C.DATALOADER.num_workers = 12
_C.DATALOADER.datapath = "./data/rel3d/c_0.9_c_0.1.json"
_C.DATALOADER.load_img = False
_C.DATALOADER.crop = False
_C.DATALOADER.norm_data = True
_C.DATALOADER.data_aug_shift = False
_C.DATALOADER.data_aug_color = False
_C.DATALOADER.resize_mask = False
_C.DATALOADER.trans_vec = []
_C.DATALOADER.predicate_dim = 30
_C.DATALOADER.object_dim = 67
_C.DATALOADER.category_map_path = ""
_C.DATALOADER.train_valid = False 
# -----------------------------------------------------------------------------
# TRAINING DETAILS
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 200
_C.TRAIN.optimizer = "adam"
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.lr_decay_ratio = 0.5
_C.TRAIN.l2 = 0.0
_C.TRAIN.layer_decay = 1.0
_C.TRAIN.scheduler = "plateau"
_C.TRAIN.early_stop = 20
_C.TRAIN.patience = 10
_C.TRAIN.warmup = 0
_C.TRAIN.predicate_specific_training= False  
_C.TRAIN.predicate_specific_start_epoch= 3       
_C.TRAIN.predicate_specific_frequency= 1         
_C.TRAIN.predicate_specific_batches=50          
_C.TRAIN.skip_normal_training= False            



# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# -----------------------------------------------------------------------------
# Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.VISION_TRANSFORMER = CN()
_C.MODEL.VISION_TRANSFORMER.in_chans = 3  
_C.MODEL.VISION_TRANSFORMER.patch_size = 16
_C.MODEL.VISION_TRANSFORMER.embed_dim = 768  
_C.MODEL.VISION_TRANSFORMER.depth = 12
_C.MODEL.VISION_TRANSFORMER.num_heads = 16
_C.MODEL.VISION_TRANSFORMER.drop_rate = 0.0
_C.MODEL.VISION_TRANSFORMER.drop_path_rate = 0.0
_C.MODEL.VISION_TRANSFORMER.pretrained = ""
_C.MODEL.VISION_TRANSFORMER.readnet_d_hidden = 512
_C.MODEL.VISION_TRANSFORMER.glove_path = ""            
_C.MODEL.VISION_TRANSFORMER.category_map_path = ""    
_C.MODEL.VISION_TRANSFORMER.attn_drop_rate = 0.0
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# DGFormer
# -----------------------------------------------------------------------------
_C.MODEL.DGFormer = CN()
_C.MODEL.DGFormer.prompt_emb_pool = 'max-in-roi'
_C.MODEL.DGFormer.use_attn_mask = True
_C.MODEL.DGFormer.drop = 0.0
_C.MODEL.DGFormer.readnet_dropout= 0.0
# -----------------------------------------------------------------------------



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
