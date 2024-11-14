from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Tuple
import os
import ml_collections
from ml_collections.config_dict import ConfigDict
import yaml
import io
import json
from pathlib import Path

def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module) -> int:
    """

    Args:
        model (torch.nn.Module): input models
    Returns:
        int: number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)

def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(
        dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def set_seed(seed: int, device='cpu'):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cupy.random.seed(seed)

    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0

def load_model_state(filepath: str, device='cpu'):
    """
    Robust function to load model state dictionaries and handle GPU/CPU transitions.
    
    Args:
        filepath (str): Path to the saved model file
        device (str): Target device ('cpu' or 'cuda')
    
    Returns:
        dict: Model state dictionary mapped to the specified device
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Helper function to handle GPU to CPU transition
    def gpu_to_cpu_state_dict(state_dict):
        """Convert GPU state dict to CPU state dict."""
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu()
            else:
                cpu_state_dict[key] = value
        return cpu_state_dict

    try:
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                try:
                    # First try loading with torch
                    state_dict = torch.load(
                        f, 
                        map_location=lambda storage, loc: storage.cpu() if device == 'cpu' else storage
                    )
                except (RuntimeError, pickle.UnpicklingError):
                    # If that fails, try regular pickle
                    f.seek(0)
                    state_dict = pickle.load(f)
                    if isinstance(state_dict, dict):
                        state_dict = gpu_to_cpu_state_dict(state_dict)
                        
        elif filepath.suffix == '.pt':
            state_dict = torch.load(
                filepath,
                map_location=lambda storage, loc: storage.cpu() if device == 'cpu' else storage
            )
            
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        return state_dict
        
    except Exception as e:
        raise RuntimeError(f"Error loading model state from {filepath}: {str(e)}")

def load_and_initialize_model(model, filepath: str, device='cpu'):
    """
    Load state dict and initialize model in one go.
    
    Args:
        model: PyTorch model instance
        filepath (str): Path to the saved model file
        device (str): Target device ('cpu' or 'cuda')
    
    Returns:
        model: Initialized model
    """
    state_dict = load_model_state(filepath, device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


def get_experiment_dir(config):
    exp_dir = './numerical_results/{dataset}/algo_{gan}_G_{generator}_D_{discriminator}_n_lag_{n_lags}_{seed}_comment_{comment}'.format(
        dataset=config.dataset, gan=config.gan_algo, generator=config.generator,
        discriminator=config.discriminator, n_lags=config.n_lags, seed=config.seed, comment=config.comment)
    os.makedirs(exp_dir, exist_ok=True)
    if config.train and os.path.exists(exp_dir):
        print("WARNING! The model exists in directory and will be overwritten")
    config.exp_dir = exp_dir


def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def loader_to_cond_tensor(dl):
    x_tensor = []
    y_tensor = []
    for x, y in dl:
        x_tensor.append(x)
        y_tensor.append(y)

    return torch.cat(x_tensor), torch.cat(y_tensor)


def load_config_ml_col(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config

def load_config(file_dir: str):
    with open(file_dir) as file:
        config = yaml.safe_load(file)
    return config

def convert_config_to_dict(config):
    """
    Conert nested ConfigDicts into dicts
    Parameters
    """
    if isinstance(config, ConfigDict):
        config = dict(config)
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = convert_config_to_dict(value)
    return config