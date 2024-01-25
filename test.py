import argparse
import torch
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.transforms import functional
import cv2
from PIL import Image
import glob

# Import Self Defined Modules
from utils.datasets import BioClassify
from model.models import InceptionV3Classifier  # Import your classification model


def inference(dataloader, model):
    preds = []
    for data, label in dataloader:
        data = Variable(data.permute(0, 3, 1, 2).cuda())
        output = model(data)
        preds.extend(output)
    # pred = torch.argmax(pred, torch.argmax(pred, dim=1).squeeze().cpu().numpy())

    print(data.shape, output.shape)
    return preds

def test(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # Set model
    model = InceptionV3Classifier()
    model.cuda()

    # Load pre-trained weights
    checkpoint = torch.load(args.weights_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load Dataset
    dataset = BioClassify(paths={"data": args.data_path})
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Run Inference
    preds = inference(dataloader, model)

    # print(preds.shape)

def parse_args():
    parser = argparse.ArgumentParser()

    # Inference parameters
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--num_workers", type = int, default = 1)

    # Paths
    parser.add_argument("--data_path", type = str, default = "")
    parser.add_argument("--weights_path", type = str, required=True)

    # Miscellanous
    parser.add_argument("--num_classes", type = int, default = 2)
    parser.add_argument("--fold", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)