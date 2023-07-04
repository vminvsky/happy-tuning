# Finetuning models
---

This repository includes code to finetune encoder-only Huggingface language models. It is mostly using PyTorch and Huggingface's Transformers library. 

This work currently has a few implemented functionalities. And a few still to be written.

- [x] Mulit-class and multi-label classification.
- [x] Training on synthetic data and evaluating on real data. 
- [ ] Threshold detection for multi-label classification. 

## Data

### Training dataset

All training datasets should be located in `data/{task}/train.json`, `data/{task}/test.json`, `data/{task}/{synthetic_name}.json` (if augmented exists).

### Dataloader

A dataloader should be built in ```src/dataloaders```. See the files their for examples.

In the general class ```AbstractDataset```in ```src/dataloaders/dataloader.py``` you can define the needed **token size**. This should be chosen depending on GPU requirements. 

## Config files

The magic of this repository happens in the config files.

```yaml
for_synthetic: False
multi_label: True
use_augmented_data: False

dataset: ubi_topics # can be 'sarcasm', 'sentiment', 'ubi_topics'
approach: taxonomy # can be 'simple', 'grounded', 'taxonomy', 'grounded_norewrite'
augmentation_model: gpt-3.5-turbo # can be gpt-3.5-turbo or gpt-4
```