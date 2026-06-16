import config_pixpro as config
import os

average_subcort_volume_path ="<EXTERNAL: average subcortical volume CSV (Result_PEM/avg_vol_subcort_regions.csv)>"
# Checkpoint Path:

experiment_name ="base"
experiment_prefix ='epoch_7'
config_path = os.path.join(config.pixpro_config_folder,experiment_name)
checkpoint_epoch_7 = os.path.join(config.model_weight_dir, experiment_name, "base_epoch=07.ckpt")

# Embedding
#------------- epoch 7----------------
result_dir_path_revised_epoch7= os.path.join(config.result_dir, experiment_name, experiment_prefix)
os.makedirs(result_dir_path_revised_epoch7, exist_ok =True)

discovery_embedding__dir = os.path.join(config.persample_embed_dir,experiment_name, 'LINEAR_MNI_discovery_combined')
replication_embedding__dir =os.path.join(config.persample_embed_dir,experiment_name, 'LINEAR_MNI_replication_combined')



# subjecrt dictionary
subject_dict_discovery = os.path.join(config.region_subject_dir, experiment_name, 'discovery_combined_epoch_7')
subject_dict_replication = os.path.join(config.region_subject_dir, experiment_name, 'replication_combined_epoch_7')


# persample region embeds

persample_region_embeds_dir = os.path.join(result_dir_path_revised_epoch7, 'persample_embedding')
os.makedirs(persample_region_embeds_dir, exist_ok =True)

