from dataloaders.dataloader import AbstractDataset
from pathlib import Path
import pandas as pd

class UBITopicDataset(AbstractDataset):
    def __init__(
        self,
        path,
        tokenizer,
        labels = ['Living costs', 'Data analysis and research', 'Education and family',
       'Non-UBI government welfare programs', 'Budget and finance',
       'Economic systems', 'Labor wages and work conditions',
       'Public services and healthcare', 'Money and inflation',
       'Politics and elections', 'Global affairs', 'Automation and jobs',
       'Taxes', 'Political affiliations', 'Business and profit',
       'None of the above'],
        is_augmented: bool = False,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        data = pd.read_json(path, orient="records").drop("id", axis=1)
        
        self.label_names = labels 
        self.num_labels = len(labels)
        self.max_length = max_length
        
        self.label2idx = {label: i for i, label in enumerate(self.label_names)}
        self.tokenizer = tokenizer 
        
        self.labels = data["labels"]
        self.texts = data["text"]