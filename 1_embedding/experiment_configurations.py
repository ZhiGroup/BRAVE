
'''This is base configuration for model version 4. Bugs in models from previous versions are fixed.
'''

import torch
import os
try:

    from . import config


except:

    import config


# <patch_size>_<sample size>_<no_positive>_<local_loss_fract>.pt
root_path = config.root_path()


class base_config_model_4(object):
    def __init__(self):
        # ----------------- Model Architecture-------------------------------------------------
        self.apply_z_stride = [2, 2, 2, 2]
        self.patch_size = (96, 96, 96)
        self.patch_size = (96, 96, 96)
        self.embedding_dim = 128

        # ----------------- self supervised training-------------------------------------------------
        self.n_pos_voxel = 50
        self.prob_overlapped_pairs = 1.0
        self.fg_pct = 0.8
        self.radius = 3
        self.local_loss_fract = 0.5
        self.los_temp = 0.5
        self.factor_n_negative_samples = 40

        # ----------------- Dataset-----------------------------------------------------------------------
        self.n_samples = -1

        # ----------------- lr-----------------------------------------------------------------------
        self.lr = 0.0005  # *0.1 #0.5*0.00010310714147574938, # start with this one. then reduce according to loss
        # schedular
        self.use_scheduler = False
        # lambdaLR
        self.sche_lamda = False
        self.lr_lambda = lambda epoch: 1*0.5*0.00010310714147574938

        self.weight_decay = 0.008

        # ----------------- training params-----------------------------------------------------------------------
        self.n_batch = 5

        self.epochs = 2000

        self.min_loss_red = 0.01

        self.perform_val_iter_cycle = 10
        self.patience = 300

        self.gpu_id = "cuda:4"
        self.device = torch.device(f'{self.gpu_id }')
        # ------------- Weights-----------------------------------------------------------------------------------

        self.weight_prefix = 'model_4'
        self.do_epoch_wise_evaluation = False

        # checkpoint:
        self.do_check_point = False
        # there is no file with such name. it should be set in specific experiment
        self.last_chk_point_name = 'model_4_base'

    @property
    def weight_name(self):
        # return f"{self.weight_prefix}_lr_{self.lr}_{self.patch_size[0]}_{self.patch_size[1]}_{self.patch_size[2]}_s_{self.n_samples}_p_{self.n_pos_voxel}_llf{self.local_loss_fract}.pt"
        return f"{self.weight_prefix}"
# ----------------- Tensorboard-----------------------------------------------------------------------

    @property
    def tensorboard_folder_name(self):
        return f'{self.weight_prefix}'


def print_conf(conf_obj):

    rep = ""
    for k, v in conf_obj.__dict__.items():
        rep += f"{k}=>{v}\n"
    # get the property set by @property decorator
    all_properties = set(f for f in dir(conf_obj) if "__" not in f)
    original_properties = set(list(conf_obj.__dict__.keys()))

    remaining_properties = all_properties.difference(original_properties)

    for f in remaining_properties:
        rep += f"{f}=>{getattr(conf_obj,f)}\n"
    return rep


class conf_model_4_exp_3(base_config_model_4):
    '''We will hyper parameters.
        1. change # negative samples to 10K
        2. Change #global to n_batch-1

    '''

    def __init__(self):

        super(conf_model_4_exp_3, self).__init__()
        # ----------------- self supervised training-------------------------------------------------
        self.model_name = 'voxel_infonce_encoder'
        self.radius = 3
        self.embedding_dim = 128
        self.memory_bank_size = 2500
        # 0.5# initally was 0.5. after one epoch, we made it asymmetrical
        self.local_loss_fract = 0.5
        self.factor_n_negative_samples = 1
        self.los_temp = 0.3
        self.los_temp_instance = 0.4
        self.n_batch = 3
        self.momentum = 0.90
        self.reconstruction_loss = True
        # ----------------- lr-----------------------------------------------------------------------
        self.lr = .005  # 6.25e-05 #0.0005*.1#*.1*.05 #1.5625e-05  #0.0005#*0.1*0.1#*0.1*0.1
        self.use_scheduler = False
        self.pct_start = 0.3
        self.div_factor = 80
        self.final_div_factor = 1000
        # ----------------- training params-----------------------------------------------------------------------
        # we are checking each iteration. save weights iteration wise
        self.perform_val_iter_cycle = 1
        self.patience = 10
        self.use_gradient_clipping = False
        self.gradient_clipping_value = 1.0
        self.max_epoch = 40

        # --------------------------- Dataset------------
        self.num_instances = 4586
        self.train_img_dir = os.path.join(
            root_path, "data", "raw_data", "ae_train")
        self.val_img_dir = os.path.join(
            root_path, "data", "raw_data", "ae_val")
        self.test_img_dir = os.path.join(
            root_path, "data", "raw_data", "T1_orig", "test")
        # ------------- Weights-----------------------------------------------------------------------------------

        self.weight_prefix = 'voxel_encoder'
        self.sub_folder = 'voxel_infonce'

        # Resuming from a checkpoint is off by default; set do_check_point and
        # point last_chk_point_name at a file under embedding.weights_dir.
        self.do_check_point = False
        self.last_chk_point_name = '<checkpoint>.pt'

        # ----------------- Dataset-----------------------------------------------------------------------
        self.n_samples = -1


