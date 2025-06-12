import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import string

class MyDataset(Dataset):
    def __init__(
        self, 
        dataset_path,
        split="train"
    ):
        self.dataset = load_dataset("parquet", data_files={
            f'{split}': f'{dataset_path}/{split}.parquet',
        })[split]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, text = self.dataset[idx]["image"].convert("RGB"), self.dataset[idx]["text"]
        # if random.randint(0, 1):
        #     text = text[:(i := random.randint(0, len(text) - 1))] + random.choice(string.ascii_letters + string.digits) + text[i+1:]
        return image, text

def pil_collate_fn(temp):
    return temp

if __name__ == "__main__":
    images_path = "./data/data"
    dataset = MyDataset(images_path)
    data_loader = DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=pil_collate_fn
    )
    for i in data_loader:
        print("***", len(i), i)
        input()
        # break
    # print(image[2].shape)