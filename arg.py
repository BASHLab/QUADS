import os
import argparse
import yaml
import random
from datetime import datetime
import numpy as np
import torch
from utils.utils import create_folder, read_yaml, munchify_dict, update_dict


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--start_from', 
        type=int,
        default=0,
        required=False
    )
    parser.add_argument(
        '--tag', 
        type=str, 
        required=False
    )
    parser.add_argument(
        '--ckpt_name', 
        type=str, 
        required=False
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        required=False
    )
    args = parser.parse_args()
    config_path = os.path.join("./config", args.config)
    config = read_yaml(config_path)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "seed" in config:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if config["device"] == "cuda":
            torch.cuda.manual_seed(config["seed"])
    
    config = update_dict(config, vars(args))
    mode = config["mode"]
    if mode == "train":
        if "tag" in config:
            if config["start_from"] == 0:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                name = f"{config['tag']}_{config['seed']}_{current_time}"
                config_path = os.path.join(
                    "./log",
                    name
                )
                config["config_path"] = config_path
                create_folder(config)
            else:
                if "log_dir" not in config:
                    raise ValueError(f"Please specify log_dir to restart training from epoch {config['start_from']}")
                config["config_path"] = config["log_dir"]
                ckpt_path = os.path.join(config["log_dir"], f"{'{:02d}'.format(config['start_from'])}_ckpt.pt")
                config["ckpt_path"] = ckpt_path

            config["shuffle"] = True
        else:
            raise ValueError("Please specify tag to start training")
    elif mode=="test":
        config["shuffle"] = False
        if "log_dir" not in config:
            raise ValueError("Please specify log_dir to evaluate")
        
        if "ckpt_name" not in config:
            raise ValueError("Please specify ckpt_name to evaluate")
        else:
            ckpt_path = os.path.join(
                config["log_dir"], f"{config['ckpt_name']}_ckpt.pt"
            )
            config["ckpt_path"] = ckpt_path
    else:
        raise NotImplementedError(f"{config['mode']} not implemented yet")

    config = munchify_dict(config)
    return config


if __name__ == "__main__":
    get_config()
