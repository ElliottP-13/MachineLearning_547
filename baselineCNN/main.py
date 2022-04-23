import os.path
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.nn import init

from networks import *

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument('-d', '--dataset', type=str, help='Path to dataset',
                    default='/mnt/data1/kwebst_data/data/GOOD_MEL_IMAGES/fold1/')

parser.add_argument('-o', '--output_dir', type=str, help='Path to save models',
                    default='/mnt/data1/kwebst_data/models/base_cnn/')

parser.add_argument('--img_size', type=int, help='sqare size of image',
                    default=224)

parser.add_argument('--batch', type=int, help='batch size',
                    default=32)

parser.add_argument('--epochs', type=int, help='number of epochs',
                    default=100)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, epochs=5, save_model_dir=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.5)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        with tqdm(enumerate(train_loader, 0), unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            running_loss = 0.0

            train_total = 0
            train_correct = 0

            for i, data in tepoch:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), accuracy=100 * (train_correct / train_total))

                if i % 100 == 499:  # log every 100
                    with open(os.path.join(save_model_dir, 'log.txt'), 'a') as f:
                        f.write(f"Epoch {epoch} -\t Step {i} -\t Training Accuracy = {100 * (train_correct / train_total)} -\t"
                                f" Running loss = {running_loss}\n")

        # Validate
        val_total = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch} -\t Validation Accuracy = {100*(val_correct / val_total)} -\t"
              f" Training Accuracy = {100*(train_correct / train_total)} -\t"
              f" Running loss = {running_loss}")

        if save_model_dir is not None:
            os.makedirs(save_model_dir, exist_ok=True)  # make the path

            torch.save(model.state_dict(), os.path.join(save_model_dir, f'modelstatedict_epoch{epoch}.model'))

            if epoch >= 5:  # get rid of old models
                os.remove(os.path.join(save_model_dir, f'modelstatedict_epoch{epoch-5}.model'))

            with open(os.path.join(save_model_dir, 'log.txt'), 'a') as f:
                f.write(f"\n\nEpoch {epoch} -\t Validation Accuracy = {100 * (val_correct / val_total)} -\t"
                        f" Training Accuracy = {100 * (train_correct / train_total)} -\t"
                        f" Running loss = {running_loss}\n\n")


if __name__ == '__main__':
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, num_workers=12, shuffle=True, pin_memory=False)

    print(f'Length of training dataset: {len(train_dataset)}')

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch, num_workers=12, shuffle=False,
        pin_memory=False)

    print(f'Length of test dataset: {len(test_dataset)}')

    model = Net4(model_name='resnet', num_classes=12)
    train(model, train_loader, test_loader, args.epochs, save_model_dir=args.output_dir)
