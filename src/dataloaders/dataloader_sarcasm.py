from dataloaders.dataloader import AbstractDataset
from pathlib import Path
import pandas as pd

class SarcasmDataset(AbstractDataset):
    def __init__(
        self,
        path,
        tokenizer,
        labels = [
            "sarcastic",
            "not-sarcastic"
        ],
        is_augmented: bool = False,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        data = pd.read_json(path, orient="records")
        
        # get text column
        text_col = "augmented_text" if is_augmented else "text"
        self.label_names = labels 
        self.num_labels = len(labels)
        self.max_length = max_length
            
        # initalize mapper
        self.label2idx = {label: i for i, label in enumerate(self.label_names)}
        self.tokenizer = tokenizer 
        
        self.labels = data["target"].apply(lambda x: self.label2idx[x])
        self.texts = data[text_col].apply(lambda x: x.split(":")[1] if len(x.split(":")) > 1 else x)            # in case of taxonomy
        
        