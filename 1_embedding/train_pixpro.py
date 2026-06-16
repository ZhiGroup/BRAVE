
import os
import argparse

import torch
import torch.nn.functional as F

import dataloader_pixpro
import config_loader_pixpro
import config_pixpro
import pixpro2
import dataloader_pixpro

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "1,5"

def train_experiment(experiment_config_path):


    experiment_config = config_loader_pixpro.load_config(experiment_config_path)
    print(f"experiment_config: {experiment_config}")



    model = pixpro2.get_model(experiment_config)

    datamodule = dataloader_pixpro.PixPro3DDataModule(batch_size = int(experiment_config['batch_size']),
                                                    patch_size = tuple(experiment_config['patch_size']),
                                                    min_overlap_fraction = float(experiment_config['min_overlap_fraction']),
                                                    max_overlap_fraction = float(experiment_config['max_overlap_fraction']),
                                                    )
    # --------------callbacks---------------
    ckdir_path = os.path.join(config_pixpro.model_weight_dir, experiment_config['experiment_name'])
    os.makedirs(ckdir_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath = ckdir_path,
                                        filename= f"{experiment_config['experiment_name']}_{{epoch:02d}}",
                                        save_top_k = -1,
                                        save_last = True
                                        
                                        )
    # --------------End of callbacks---------------

    #---------------- Logger-------------
    
    tb_logger = TensorBoardLogger(save_dir = "log",
                                  name = experiment_config['experiment_name'],
                                  
                                  )
    



    trainer = Trainer(max_epochs = int(experiment_config['epochs']),
                    accelerator ='gpu',
                    devices = [0,1],
                    log_every_n_steps = 10,
                    strategy='ddp',
                    callbacks =[checkpoint_callback],
                    logger = tb_logger,
                    gradient_clip_val = 1.5,
                    gradient_clip_algorithm = 'value',
                    synch_batc_norm= True,
                    )

    trainer.fit( model=model,  datamodule= datamodule, )
    
def debug_validation_step(model, batch):
        im_1, im_2, coord1, coord2, foreground_mask_im_1, foreground_mask2_im_2 = batch
        #print(f"Calling model.encoder(im_1) with im_1 shape: {im_1.shape}")
        recons_1, d3_1 = model.encoder(im_1)
        #print(f"Retuned from model.encoder(im_1) with recons_1 shape: {recons_1.shape}, d3_1: {d3_1.shape}")
        proj_1 = model.projector(d3_1)
        #print(f"Retuned from  model.projector(d3_1) proj_1 shape: {proj_1.shape}")        
        pred_1 = F.normalize(model.featprop(proj_1), dim=1)
        #print(f"Retuned from  F.normalize(model.featprop(proj_1), dim=1) pred_1 shape: {pred_1.shape}")     
         
        recons_2, d3_2 = model.encoder(im_2)
        proj_2 = model.projector(d3_2)
        pred_2 = F.normalize(model.featprop(proj_2), dim=1)

        with torch.no_grad():
            model._momentum_update_key_encoder()
            _, d3_1_k = model.encoder_k(im_1)
            proj_1_k = F.normalize(model.projector_k(d3_1_k), dim=1)

            _, d3_2_k = model.encoder_k(im_2)
            proj_2_k = F.normalize(model.projector_k(d3_2_k), dim=1)
        #print(f"Calling from  regression_loss(pred_1, proj_2_k, coord1, coord2, model.pixpro_pos_ratio)\npred_1 shape: {pred_1.shape},proj_2_k: {proj_2_k.shape}, coord1: {coord1.shape},coord2: {coord2.shape} ")         
        consistancy_loss = pixpro2.regression_loss(pred_1, proj_2_k, coord1, coord2, model.pixpro_pos_ratio) + \
            pixpro2.regression_loss(pred_2, proj_1_k, coord2,
                            coord1,  model.pixpro_pos_ratio)

        recon_loss = model.compute_recons_loss(
            im_1, recons_1,  foreground_mask_im_1) + model.compute_recons_loss(im_2, recons_2,  foreground_mask2_im_2)

        loss = consistancy_loss + recon_loss
        print()    
    
    
def main():
    parser = argparse.ArgumentParser(description='Pixpro')
    parser.add_argument('--config_path', type = str, default = 'base_config.yaml')
    
    args = parser.parse_args()
    experiment_config_path = os.path.join(config_pixpro.pixpro_config_folder,args.config_path)
    print(f"Cofig file choose was : {experiment_config_path}")
    train_experiment(experiment_config_path)
if __name__ == '__main__':

    # experiment_config = config_loader_pixpro.load_config(experiment_config_path)
    # print(f"experiment_config: {experiment_config}")
    # model = pixpro2.get_model(experiment_config)

    # dl = dataloader_pixpro.get_dataloader(batch_size=3,
    #                type_dataloader='train',
    #                shape=(96, 96, 96),
    #                min_overlap_fraction=0.2, max_overlap_fraction=0.8)
    # for batch in dl:
    #     debug_validation_step(model, batch)
    #     break
    
    main()
    
