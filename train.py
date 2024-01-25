import argparse
import torch
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import nn

# Import Self Defined Modules
from utils.datasets import BioClassify
from model.models import InceptionV3Classifier

def train(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # Set datasets
    train_dataset = BioClassify(paths={"data": args.train_data})
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set model
    model = InceptionV3Classifier(num_classes=args.num_classes)  # Change to your classification model
    model.cuda()

    # Set loss function
    criterion = nn.functional.cross_entropy  # Change to your classification loss function

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set Paths
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set tensorboard writer
    run_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, "loss", f"{run_time}_{str(args.fold)}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Track Metrics
    train_loss = []
    valid_loss = []

    # Train
    for epoch in range(args.epochs):
        model.train()
        loss_meter = []
        for data, label in tqdm(train_loader):
            data = data.permute(0, 3, 1, 2)
            data = Variable(data.cuda())
            label = Variable(label.cuda())
            optimizer.zero_grad()
            output = torch.as_tensor(model(data)[0])
            loss = criterion(output, label)
            loss_meter.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_avg = np.mean(loss_meter)
        train_loss.append(loss_avg)
        writer.add_scalar("Loss/train", loss_avg, epoch)
        print(f"(Train) Epoch: {epoch}/{args.epochs} Loss: {loss_avg}")

        if epoch % args.save_every == 0:
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(args.save_path, f"{run_time}_{str(args.fold)}_model.pth"))

            # Validate (for classification task)
            if args.valid_data:
                val_loss = validate(args, model, criterion)
                valid_loss.append(val_loss)
                print(f"(Valid) Epoch: {epoch}/{args.epochs} Loss: {val_loss}")
                writer.add_scalar("Loss/valid", val_loss, epoch)

                # Save best model based on accuracy
                if val_loss == np.min(valid_loss):
                    torch.save(model.state_dict(), os.path.join(args.save_path, f"{run_time}_{str(args.fold)}_model_best.pth"))

def validate(args, model, criterion):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set datasets
    valid_dataset = BioClassify(paths={"data": args.valid_data})
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Run validation
    model.eval()
    loss_meter = []
    with torch.no_grad():
        for data, label in tqdm(valid_loader):
            data = data.permute(0, 3, 1, 2)
            data = Variable(data.to(device))
            label = Variable(label.to(device))
            output = torch.as_tensor(model(data))
            loss = criterion(output, label)
            loss_meter.append(loss.item())
    
    return np.mean(loss_meter)

def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--save_every", type = int, default = 5)
    parser.add_argument("--num_workers", type = int, default = 5)

    # Paths
    parser.add_argument("--train_data", type = str, default = "")
    parser.add_argument("--valid_data", type = str, default = "")
    parser.add_argument("--save_path", type = str, default = "./checkpoints/")
    parser.add_argument("--log_dir", type = str, default = "./logs/")

    # Miscellanous
    parser.add_argument("--num_classes", type = int, default = 2)
    parser.add_argument("--fold", type=int, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)