from typing import Dict, Iterable, List, Union
import torch
from pathlib import Path
import pandas as pd
import json

def parse_output(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for i, item in enumerate(input_list):
        item = item.lstrip(
            "0123456789. "
        )  # remove enumeration and any leading whitespace
        if item:  # skip empty items
            output_list.append(item)
    return output_list

def get_device() -> torch.device:
    """Get device."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device

def read_json(path: Path) -> List[Dict[str, Union[str, int]]]:
    with open(path, "r") as f:
        data: List[Dict[str, Union[str, int]]] = json.load(f)
    return data

def save_json(path: Path, container: Iterable) -> None:
    """write dict to path."""
    print(f"Saving json to {path}")
    with open(path, "w") as outfile:
        json.dump(container, outfile, ensure_ascii=False, indent=4)
        
        
def create_synthetic_real(synthetic_path, real_path, pos_label, balance=False):
    synthetic = pd.read_json(synthetic_path, orient="records")
    real = pd.read_json(real_path, orient="records")
    
    if balance:
        other_class_count = real[real["labels"]==pos_label].count()
        pos_class_count = real.count()
        
    synthetic = synthetic[synthetic["labels"]==pos_label].sample(pos_class_count)
    real = real[real["labels"]!=pos_label]
    
    concattenated = pd.concat([synthetic, real], axis=0)
    
    return concattenated