from huggingface_hub import snapshot_download
from datasets import load_dataset
import base64
from PIL import Image, ImageDraw
import io
import os

# Download DocSynth300K

def download(name, save_dir):
    os.path.exists(save_dir) or os.makedirs(save_dir)
    snapshot_download(repo_id=name, local_dir=save_dir, repo_type="dataset", resume_download=True)

def show_data(data_dir=None):
    dataset = load_dataset("parquet", data_files={
        'train': './data/train.parquet',
        'val': './data/train.parquet',
        'test': './data/train.parquet'
    })["test"]
    for i in dataset:
        print(i)
        break
if __name__ == "__main__":
    # download("Teklia/IAM-line", "./")
    show_data()
