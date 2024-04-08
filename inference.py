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
from part_a_train import CNN
# Initialize an instance of your model

device = torch.device("cuda:0")
model = CNN().to(device)

# Load the model state dict
model.pth = '/home/bincy/A2/CS-6910-A2/mymodel.pth'
model.load_state_dict(torch.load(model.pth))

# Move the model to the CUDA device after loading the state dict
model = model.to(device)
wandb.login() 
wandb.init("CS6910 A2",name="testing")





### preparing data
def prepare(batch_size=32):
    print("preparing data")
    common_transforms = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

   
  
    test_transform = transforms.Compose(common_transforms)
   
    test_data_path = '/home/bincy/A2/CS-6910-A2/inaturalist_12K/val'

    test_dataset = datasets.ImageFolder(test_data_path, transform=test_transform)
   
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return  testloader
def evaluation(dataloader,model):
    total, correct = 0, 0
    device = torch.device("cuda:0")
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(outputs, labels)
                
        _, pred = torch.max(outputs.data, 1)
        
        
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
    return 100 * correct / total,loss.item()



def get_all_preds(model, loader):
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
def plot_confusion_matrix(labels, preds, classes):
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
        wandb.log({"confusion_matrix": wandb.Image(fig)})
def imshow(img):
    npimg = img.cpu().numpy()  # Move tensor to CPU before converting to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


print("create loaders")
device = torch.device("cuda:0")
testloader=prepare(batch_size=32)
model.eval()
best_model_accuracy,best_model_loss=evaluation(testloader,model)

print('validation acc from best model: %0.2f,val loss: %0.2f'%(best_model_accuracy,best_model_loss))


true_labels, predictions = get_all_preds(model, testloader)  
classes = testloader.dataset.classes 

plot_confusion_matrix(true_labels, predictions, classes)

#class_names = ('Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')
plt.show()

def show_random_predictions(test_loader, model, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Select 30 random images from the test loader
    random_indices = random.sample(range(len(test_loader.dataset)), 30)
    images, labels = [], []
    for idx in random_indices:
        image, label = test_loader.dataset[idx]
        images.append(image)
        labels.append(label)

    # Prepare the images for visualization
    images = torch.stack(images).to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # Create a grid of images and their corresponding predictions
    fig, axs = plt.subplots(10, 3, figsize=(15, 30))
    fig.subplots_adjust(hspace=0.5)
    
    for i in range(10):
        for j in range(3):
            idx = i * 3 + j
            image = images[idx].cpu().permute(1, 2, 0)
            pred_class = class_names[preds[idx].item()]
            actual_class = class_names[labels[idx]]
            
            axs[i, j].imshow(image)
            axs[i, j].axis('off')
            axs[i, j].set_title(f'Predicted: {pred_class}, Actual: {actual_class}')

    plt.show()
    wandb.log({"image_grid": wandb.Image(fig)})
    
    # Close the figure to release memory
    plt.close(fig)

# Assuming `testloader` and `model` are already defined
class_names = ('Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')
show_random_predictions(testloader, model, class_names)