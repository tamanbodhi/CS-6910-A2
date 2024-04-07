import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import wandb
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse
class PreModel:
    def prepare(self,batch_size=16, use_data_augmentation=True):
    # Common normalization and resize operations
        common_transforms = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]

        # Define transformations for the training data with optional augmentation
        if use_data_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Example augmentation
                transforms.RandomRotation(10),  # Example augmentation
                *common_transforms,
            ])
        else:
            train_transform = transforms.Compose(common_transforms)

        # Transformations for validation and test data (no augmentation)
        test_transform = transforms.Compose(common_transforms)

        train_data_path = '/home/bincy/A2/CS-6910-A2/inaturalist_12K/train'
        test_data_path = '/home/bincy/A2/CS-6910-A2/inaturalist_12K/val'

        # Load the training dataset with the train_transform
        full_train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)

        # Split the full training dataset into training and validation subsets
        validation_size = 0.2
        stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=42)
        train_indices, validation_indices = next(stratified_splitter.split(np.array(full_train_dataset.targets), np.array(full_train_dataset.targets)))

        # Create subsets for training and validation
        train_subset = Subset(full_train_dataset, train_indices)
        validation_dataset = datasets.ImageFolder(train_data_path, transform=test_transform)  # Reload with test_transform
        validation_subset = Subset(validation_dataset, validation_indices)

    
        test_dataset = datasets.ImageFolder(test_data_path, transform=test_transform)

        # Create DataLoaders for training, validation, and test datasets
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validationloader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return trainloader, validationloader, testloader
    def evaluation(self,dataloader, model):
        total, correct = 0, 0
        device = torch.device("cuda:0")
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total

    def get_all_preds(self,model, loader):
        all_preds = []
        all_labels = []
        device = torch.device("cuda:0")
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        return all_labels, all_preds
    def plot_confusion_matrix(self,labels, preds, classes):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    def fitresnet(self):
        num_classes=10
        batch_size=16
        trainloader,validationloader,testloader=self.prepare(batch_size)
        device = torch.device("cuda:0")
        resnet = models.resnet18(pretrained=True)
        print(resnet)
        
        for param in resnet.parameters():
            param.requires_grad = False
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        for param in resnet.parameters():
            if param.requires_grad:
                print(param.shape)
        dropout_rate = 0.0  # Example dropout rate, adjust as needed
        resnet.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
        )
        print(resnet)
        resnet = resnet.to(device)
        
        resnet = resnet.to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.SGD(resnet.parameters(), lr=0.01)
        loss_epoch_arr = []
        max_epochs = 1

        min_loss = 1000

        n_iters = np.ceil(10000/batch_size)

        for epoch in range(max_epochs):
            resnet.train()
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                opt.zero_grad()

                outputs = resnet(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()
                
                if min_loss > loss.item():
                    min_loss = loss.item()
                    best_model = copy.deepcopy(resnet.state_dict())
                    print('Min loss %0.2f' % min_loss)
                
                if i % 100 == 0:
                    print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))
                    
                del inputs, labels, outputs
                torch.cuda.empty_cache()
                
            loss_epoch_arr.append(loss.item())
            resnet.eval()   
            print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
                epoch, max_epochs, 
                self.evaluation(testloader, resnet), self.evaluation(trainloader, resnet)))
            
        path_to_save='/home/bincy/A2/CS-6910-A2/model.pth'
        torch.save(resnet.state_dict(), path_to_save)
        
        plt.plot(loss_epoch_arr)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss vs epoch')
        
        true_labels, predictions = self.get_all_preds(resnet, testloader)  
        classes = testloader.dataset.classes 

        self.plot_confusion_matrix(true_labels, predictions, classes)
        plt.show()
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity',default="bincyantonym")
    parser.add_argument('--wandb_project',default="CS6910 A2")
    mm=PreModel()
    
    mm.fitresnet()