import glob
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import pandas as pd

class BioClassify(Dataset):
    def __init__(self, paths = {"data": None}):
        self.data = pd.read_csv(paths["data"])

    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        img = cv2.imread(self.data.loc[idx, "path"])
        label = self.data.loc[idx, "label"]

        img = torch.as_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        label = torch.as_tensor(label)

        return img, label