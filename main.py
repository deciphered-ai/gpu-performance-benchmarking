'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import time 
import argparse

from pathlib import Path

from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'  
else:
    print("Sorry! torch could not detect a GPU.\n The code will exit now...")
    exit(0)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

def sample_models_for_images(model_name, net,csvfile):
    # Taken with love from kuangliu's github repo 
    # https://github.com/kuangliu/pytorch-cifar

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if os.path.exists("data/cifar-10-batches-py"):
        download = False
    else:
        donwload = True

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)


    for epoch in [1,2]:
        start_train = time.time()
        train(epoch)
        test(epoch)
        scheduler.step()
        if epoch ==2:
            print("Time to train {0} for 1 epoch: {1} seconds".format(model_name,str(time.time() - start_train)))
            csvfile.write("{0}, {1}\n".format(model_name, str(time.time() - start_train)))

def bert_for_text_classification(csvfile):
    train_texts, train_labels = read_imdb_split('data/aclImdb/train')
    test_texts, test_labels = read_imdb_split('data/aclImdb/test')

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in [1,2]:
        start_train = time.time()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
        if epoch == 2:
            print("Time to train {0} for 1 epoch: {1} seconds".format("distilled bert",str(time.time() - start_train)))
            csvfile.write("{0}, {1}\n".format("distilled bert", str(time.time() - start_train)))

if __name__ == "__main__":
    # Model
    print('==> Starting tests for image data..')

    csvfile = open("benchmark.csv","w")
    csvfile.write("model_name, time for one epoch\n")

    various_nets = {'vgg':VGG('VGG19'), 
            'ResNet':ResNet18(), 
            'DenseNet121':DenseNet121(), 
            'ResNeXt29_2x64d':ResNeXt29_2x64d(), 
            'MobileNetV2':MobileNetV2(),  
            'ShuffleNetV2':ShuffleNetV2(1),  
            'EfficientNetB0':EfficientNetB0()}
    for name,net in various_nets.items():
        sample_models_for_images(name, net,csvfile)
    
    print('==> Starting tests for sequential data..')
    bert_for_text_classification(csvfile)
    csvfile.close()