# train.py
import os
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True,
                   help='Directory with train/val subfolders (ImageFolder structure)')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--out', type=str, default='model.pth')
    return p.parse_args()

def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(data_dir/'train', transform=train_transforms)
    val_ds = datasets.ImageFolder(data_dir/'val', transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v2(pretrained=True)
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, len(train_ds.classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct/total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds==labels).sum().item()
                val_total += imgs.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'classes': train_ds.classes
            }, args.out)
            print("Saved best model")

if __name__=='__main__':
    main()
