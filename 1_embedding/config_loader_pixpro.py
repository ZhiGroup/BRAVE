import yaml
import json

def load_config(config_file):
    """
    Load configuration from a YAML or JSON file.
    If a value is missing in the config file, defaults will be used.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration with defaults applied.
    """
    
    #default configuration 
    
    default_configs = {
        "pixpro_p": 1.0,
        "pixpro_momentum": 0.99,
        "pixpro_pos_ratio": 0.7,
        "pixpro_ins_loss_weight": 1.0,
        "batch_size": 3,
        "learning_rate": 0.001,
        "epochs": 100,
        "embed_dim": 128,
        'lr': 1e-3,
        'weight_decay': 1e-5
        
    }
    if config_file:
        print(f"loading configs from {config_file}")
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_file.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError('Unsupported configuration file format. Use YAML or Json')
    else:
        raise ValueError('no config file path was given')
    
    default_configs.update(config)
    
    return default_configs
    