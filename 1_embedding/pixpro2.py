import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import lightning.pytorch as pl


from torch.distributed import get_world_size
import resent_pixpro2
import dataloader_pixpro


class Identity(pl.LightningModule):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1_3d(in_planes, out_planes):
    """1x1 convolution for 3D"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP3d(pl.LightningModule):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP3d, self).__init__()
        self.linear1 = conv1x1_3d(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm3d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1_3d(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

def Proj_Head(in_dim = 256, inner_dim = 4096, out_dim =256):
    return MLP3d(in_dim, inner_dim, out_dim)
    
def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP3d(in_dim, inner_dim, out_dim) 
    
def regression_loss(q, k, coord_q, coord_k,  pos_ratio=0.7):
    """
        q, k: N * C * D * H * W (feature maps)
        coord_q, coord_k: N * 6 (x_upper_left, y_upper_left, z_upper_left, x_lower_right, y_lower_right, z_lower_right)

        pos_ratio: Threshold ratio for positive pairs

    """

    N, C, D, H, W = q.shape
    # Flatten spatial dimensions
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate 3D coordinate grids
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype,
                           device=coord_q.device).view(1, 1, 1, -1).repeat(1, D, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype,
                           device=coord_q.device).view(1, 1, -1, 1).repeat(1, D, 1, W)
    z_array = torch.arange(0.,  float(D), dtype=coord_q.dtype,
                           device=coord_q.device).view(1, -1, 1, 1).repeat(1, 1, H, W)

    # Calculate bin sizes for query and key coordinates

    q_bin_width = ((coord_q[:, 3] - coord_q[:, 0])/W).view(-1, 1, 1, 1)
    q_bin_height = ((coord_q[:, 4] - coord_q[:, 1]) / H).view(-1, 1, 1, 1)
    q_bin_depth = ((coord_q[:, 5] - coord_q[:, 2]) / D).view(-1, 1, 1, 1)
    k_bin_width = ((coord_k[:, 3] - coord_k[:, 0]) / W).view(-1, 1, 1, 1)
    k_bin_height = ((coord_k[:, 4] - coord_k[:, 1]) / H).view(-1, 1, 1, 1)
    k_bin_depth = ((coord_k[:, 5] - coord_k[:, 2]) / D).view(-1, 1, 1, 1)

    # Calculate bin start coordinates
    q_start_x = coord_q[:, 0].view(-1, 1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1, 1)
    q_start_z = coord_q[:, 2].view(-1, 1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1, 1)
    k_start_z = coord_k[:, 2].view(-1, 1, 1, 1)

    # Calculate diagonal of the bin for normalization
    q_bin_diag = torch.sqrt(
        q_bin_width ** 2 + q_bin_height ** 2 + q_bin_depth ** 2)
    k_bin_diag = torch.sqrt(
        k_bin_width ** 2 + k_bin_height ** 2 + k_bin_depth ** 2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag).view(-1,1,1)

    # Compute center coordinates for query and key
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_q_z = (z_array + 0.5) * q_bin_depth + q_start_z
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y
    center_k_z = (z_array + 0.5) * k_bin_depth + k_start_z

    # Flatten spatial dimensions
    center_q_x = center_q_x.view(-1, D * H * W, 1)
    center_q_y = center_q_y.view(-1, D * H * W, 1)
    center_q_z = center_q_z.view(-1, D * H * W, 1)
    center_k_x = center_k_x.view(-1, 1, D * H * W)
    center_k_y = center_k_y.view(-1, 1, D * H * W)
    center_k_z = center_k_z.view(-1, 1, D * H * W)

    # Compute Euclidean distance between query and key centers
    dist_center = torch.sqrt(
        (center_q_x - center_k_x) ** 2 +
        (center_q_y - center_k_y) ** 2 +
        (center_q_z - center_k_z) ** 2
    ) / max_bin_diag
    # Create a mask for positive pairs
    pos_mask = (dist_center < pos_ratio).float().detach()

    # Compute logits using dot product
    logit = torch.bmm(q.transpose(1, 2), k)

    # Compute the loss
    loss = (logit * pos_mask).sum(-1).sum(-1) / \
        (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


class PixPro3D(pl.LightningModule):
    def __init__(self, base_encoder, args):
        super(PixPro3D, self).__init__()
        self.save_hyperparameters()
        self.embed_dim = args['embed_dim']
        self.pixpro_p = args['pixpro_p']
        self.pixpro_momentum = args['pixpro_momentum']
        self.pixpro_pos_ratio = args['pixpro_pos_ratio']
        self.pixpro_clamp_value = args['pixpro_clamp_value']
        self.pixpro_ins_loss_weight = args['pixpro_ins_loss_weight']
        self.pixpro_transform_layer = args['pixpro_transform_layer']
        if 'loss_weight_reconst' in args:
            self.loss_weight_reconst = args['loss_weight_reconst']
        else:
            self.loss_weight_reconst = 1.0
        self.args = args

        self.encoder = base_encoder  # U-Net3D for encoder-decoder
        self.projector = MLP3d(256, 4096, 256)

        self.encoder_k = base_encoder  # Momentum encoder
        self.projector_k = MLP3d(256, 4096, 256)

        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1_3d(
                in_planes=self.embed_dim, out_planes=self.embed_dim)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP3d(
                in_dim=self.embed_dim, inner_dim=self.embed_dim, out_dim=self.embed_dim)
        else:
            raise NotImplementedError

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Freeze momentum encoder

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        self.projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        self.projector_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        
        # instance
        if self.pixpro_ins_loss_weight > 0:
            print(f"Using instance Loss")
            self.projector_instance = Proj_Head()
            self.projector_instance_k = Proj_Head()
            self.predictor = Pred_Head()
            
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
                
                
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

            self.avgpool = nn.AvgPool3d(24, stride=1)
        
        self.recons_loss_metric = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction='none')

        self.K = None  # Will be set in setup
        self.k = None  # Will be set in setup

    def setup(self, stage=None):
        # This is called after the trainer is attached
        world_size = self.trainer.world_size if self.trainer else 1
        self.K = int(self.args['num_instances'] * 1.0 / world_size /
                     self.args['batch_size'] * self.trainer.max_epochs)
        self.k = int(self.args['num_instances'] * 1.0 / world_size /
                     self.args['batch_size'] * (self.trainer.current_epoch))

    def on_train_start(self):
        # Another hook where I can access the trainer
        world_size = self.trainer.world_size if self.trainer else 1
        self.K = int(self.args['num_instances'] * 1.0 / world_size /
                     self.args['batch_size'] * self.trainer.max_epochs)
        self.k = int(self.args['num_instances'] * 1.0 / world_size /
                     self.args['batch_size'] * (self.trainer.current_epoch))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        _contrast_momentum = 1. - \
            (1. - self.pixpro_momentum) * \
            (np.cos(np.pi * self.k / self.K) + 1) / 2.
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + \
                param_q.data * (1. - _contrast_momentum)
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + \
                param_q.data * (1. - _contrast_momentum)
                
                
        if self.pixpro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data = param_k.data *_contrast_momentum + param_q.data*(1-_contrast_momentum)

    def featprop(self, feat):
        N, C, D, H, W = feat.shape
        feat_value = F.normalize(
            self.value_transform(feat), dim=1).view(N, C, -1)

        # similarity calcualation
        feat = F.normalize(feat, dim=1)
        feat = feat.view(N, C, -1)

        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention += 1e-6
        attention = attention ** self.pixpro_p
        return torch.bmm(feat_value, attention.transpose(1, 2)).view(N, C, D, H, W)

    def compute_recons_loss(self, origianl, recon, foreground_mask):
        """
            recon, origianl: (N, 1, D,H,W)
            foreground_mask: (N, D, H, W)
        """
        # (N, 1, D,H,W) -> (N, 1, D,H,W)
        voxe_wise_loss = self.recons_loss_metric(origianl, recon)
        voxe_wise_loss = voxe_wise_loss.squeeze(1) * foreground_mask
        per_sample_loss  = voxe_wise_loss.sum(dim=(1,2,3))/ foreground_mask.sum(dim=(1, 2, 3))
        loss = per_sample_loss.mean()
        return loss
    
    def regression_loss(self, x, y):
        return -2. * torch.einsum('nc, nc->n', [x, y]).mean()
    
    
    def forward(self, x):
        """
        Forward method for inference.
        Args:
            x: Input 3D tensor (batch, channels, depth, height, width).
        Returns:
            recons: Reconstructed 3D tensor.
            feature_map: Feature map (e.g., d3).
        """
        print(f"Forward call is not called")
        recons, feature_map = self.encoder(x)
        return recons, feature_map

    def training_step(self, batch, batch_idx):
        """
            im_1, im_2: (b,1, D,H,W)
            coord1, coord2: bounding box coord for im_1, im_2. 
                            N * 6 (x_upper_left, y_upper_left, z_upper_left, x_lower_right, y_lower_right, z_lower_right)
            foreground_mask_im_1, foreground_mask2_im_2: foreground mask for img_1, img_2

        """
        im_1, im_2, coord1, coord2, foreground_mask_im_1, foreground_mask2_im_2 = batch

        recons_1,  _, d3_1 = self.encoder(im_1)

        proj_1 = self.projector(d3_1)
      
        pred_1 = F.normalize(self.featprop(proj_1), dim=1)
         
        recons_2, _, d3_2 = self.encoder(im_2)
        proj_2 = self.projector(d3_2)
        pred_2 = F.normalize(self.featprop(proj_2), dim=1)


        # instacne
        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in forward")
            proj_instance_1 = self.projector_instance(d3_1)
            pred_instacne_1 = self.predictor(proj_instance_1)
            pred_instance_1 =F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0),-1), dim = 1)

            proj_instance_2 = self.projector_instance(d3_2)
            pred_instacne_2 = self.predictor(proj_instance_2)
            pred_instance_2 =F.normalize(self.avgpool(pred_instacne_2).view(pred_instacne_2.size(0),-1), dim = 1)            
            
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, _, d3_1_k = self.encoder_k(im_1)
            proj_1_k = F.normalize(self.projector_k(d3_1_k), dim=1)

            _, _, d3_2_k = self.encoder_k(im_2)
            proj_2_k = F.normalize(self.projector_k(d3_2_k), dim=1)
        
            if self.pixpro_ins_loss_weight > 0.:
                #print(f"Using instance Loss in training step caluation")
                proj_instance_1_ng = self.projector_instance_k(d3_1_k)
                proj_instance_1_ng = F.normalize(self.avgpool(proj_instance_1_ng).view(proj_instance_1_ng.size(0),-1), dim = 1)
        
                proj_instance_2_ng = self.projector_instance_k(d3_2_k)
                proj_instance_2_ng = F.normalize(self.avgpool(proj_instance_2_ng).view(proj_instance_2_ng.size(0),-1), dim = 1)        
        
        
        consistancy_loss = regression_loss(pred_1, proj_2_k, coord1, coord2, self.pixpro_pos_ratio) + \
            regression_loss(pred_2, proj_1_k, coord2,
                            coord1,  self.pixpro_pos_ratio)

        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in training step loss calcualtion")
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) +\
                            self.regression_loss(pred_instance_2, proj_instance_1_ng) 
            
        recon_loss = self.compute_recons_loss(
            im_1, recons_1,  foreground_mask_im_1) + self.compute_recons_loss(im_2, recons_2,  foreground_mask2_im_2)

        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in training step loss calcualtion 2")
            loss = consistancy_loss + self.pixpro_ins_loss_weight* loss_instance+ self.loss_weight_reconst*recon_loss
        else:
            loss = consistancy_loss + recon_loss
        
        
        # Log losses
        self.log("train_total_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_consistency_loss", consistancy_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.pixpro_ins_loss_weight > 0.:        
            self.log("train_instance_loss", loss_instance,
                    on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
            im_1, im_2: (b,1, D,H,W)
            coord1, coord2: bounding box coord for im_1, im_2. 
                            N * 6 (x_upper_left, y_upper_left, z_upper_left, x_lower_right, y_lower_right, z_lower_right)
            foreground_mask_im_1, foreground_mask2_im_2: foreground mask for img_1, img_2

        """
        im_1, im_2, coord1, coord2, foreground_mask_im_1, foreground_mask2_im_2 = batch

        recons_1,  _, d3_1 = self.encoder(im_1)

        proj_1 = self.projector(d3_1)
      
        pred_1 = F.normalize(self.featprop(proj_1), dim=1)
         
        recons_2, _, d3_2 = self.encoder(im_2)
        proj_2 = self.projector(d3_2)
        pred_2 = F.normalize(self.featprop(proj_2), dim=1)


        # instacne
        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in validation step caluation")
            proj_instance_1 = self.projector_instance(d3_1)
            pred_instacne_1 = self.predictor(proj_instance_1)
            pred_instance_1 =F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0),-1), dim = 1)

            proj_instance_2 = self.projector_instance(d3_2)
            pred_instacne_2 = self.predictor(proj_instance_2)
            pred_instance_2 =F.normalize(self.avgpool(pred_instacne_2).view(pred_instacne_2.size(0),-1), dim = 1)            
            
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, _, d3_1_k = self.encoder_k(im_1)
            proj_1_k = F.normalize(self.projector_k(d3_1_k), dim=1)

            _, _, d3_2_k = self.encoder_k(im_2)
            proj_2_k = F.normalize(self.projector_k(d3_2_k), dim=1)
        
            if self.pixpro_ins_loss_weight > 0.:
                #print(f"Using instance Loss in validation step caluation momentum")
                proj_instance_1_ng = self.projector_instance_k(d3_1_k)
                proj_instance_1_ng = F.normalize(self.avgpool(proj_instance_1_ng).view(proj_instance_1_ng.size(0),-1), dim = 1)
        
                proj_instance_2_ng = self.projector_instance_k(d3_2_k)
                proj_instance_2_ng = F.normalize(self.avgpool(proj_instance_2_ng).view(proj_instance_2_ng.size(0),-1), dim = 1)        
        
        
        consistancy_loss = regression_loss(pred_1, proj_2_k, coord1, coord2, self.pixpro_pos_ratio) + \
            regression_loss(pred_2, proj_1_k, coord2,
                            coord1,  self.pixpro_pos_ratio)

        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in validation step caluation loss")
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) +\
                            self.regression_loss(pred_instance_2, proj_instance_1_ng) 
            
        recon_loss = self.compute_recons_loss(
            im_1, recons_1,  foreground_mask_im_1) + self.compute_recons_loss(im_2, recons_2,  foreground_mask2_im_2)

        if self.pixpro_ins_loss_weight > 0.:
            #print(f"Using instance Loss in validation step loss 2")
            loss = consistancy_loss + self.pixpro_ins_loss_weight* loss_instance+ self.loss_weight_reconst*recon_loss
        else:
            loss = consistancy_loss + recon_loss
        
        
        # Log losses
        self.log("val_total_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_consistency_loss", consistancy_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.pixpro_ins_loss_weight > 0.:        
            self.log("val_instance_loss", loss_instance,
                    on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

        








    def configure_optimizers(self):
        print("Configuring optimizers...")

        # Check all arguments
        print("Arguments:", self.args)

        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.args['lr']),
            weight_decay=float(self.args['weight_decay'])
        )

        # Get DataLoader and check steps per epoch
        train_dataloader = self.trainer.datamodule.train_dataloader()
        if train_dataloader is None:
            raise ValueError("train_dataloader is None. Ensure your datamodule is correctly implemented.")
        steps_per_epoch = len(train_dataloader)
        print(f"Steps per epoch: {steps_per_epoch}")

        # Get epochs
        epochs = self.trainer.max_epochs
        if epochs is None:
            raise ValueError("self.trainer.max_epochs is None. Ensure you have set max_epochs in your Trainer.")

        # Define OneCycleLR schedule
        lr_scheduler_config = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=float(self.args['lr']),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                pct_start=float(self.args['pct_start']),
                div_factor=int(self.args['div_factor']),
                final_div_factor=int(self.args['final_div_factor']),
            ),
            "interval": "step",
            "name": "one_cycle_lr"
        }

        print("Optimizer and scheduler configured successfully.")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}



class Args:
    def __init__(self):
        self.pixpro_p = 0.5
        self.pixpro_momentum = 0.99
        self.pixpro_pos_ratio = 0.7
        self.pixpro_clamp_value = 1e-6
        self.pixpro_ins_loss_weight = 0.1
        self.batch_size = 6
        self.lr = 1e-4
        self.epochs = 50


def get_model(args, do_inference=False, weight_path = None, device = 'cpu'):
    # encoder

    unet_encoder = resent_pixpro2.UNet3D(in_channels=1, out_channels=1)
    model = PixPro3D(unet_encoder, args)

    if do_inference:
        print(f"weight path: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        print(f"checkpoint loaded from {checkpoint}")
        model.load_state_dict(checkpoint['state_dict'])
        print(f"state_dict loaded")
        return model.encoder
    
    return model


