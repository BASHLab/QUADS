import os
import json
import yaml
import torch
from munch import Munch


def json_to_dict(file_location):
    with open(file_location, "r") as json_file:
        data_dict = json.load(json_file)
    return data_dict


def read_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def munchify_dict(raw_dict):
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            raw_dict[k] = munchify_dict(v)
    return Munch(raw_dict)


def save_ckpt(epoch, model, config):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config
        },
        os.path.join(config.config_path, f"{epoch:02d}_ckpt.pt")
    )


def load_ckpt(model, config):
    model.load_state_dict(
        torch.load(config.ckpt_path, map_location="cpu")["model"]
    )
    return model


def create_folder(config):
    os.makedirs(config["config_path"])


def update_dict(old_dict, new_dict):
    for key, value in new_dict.items():
        if key not in old_dict:
            old_dict[key] = value
    return old_dict
