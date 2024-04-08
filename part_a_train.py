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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns

def prepare(batch_size=32, use_data_augmentation=False):
   
    common_transforms = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    if use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(10), 
            *common_transforms,
        ])
    else:
        train_transform = transforms.Compose(common_transforms)

  
    test_transform = transforms.Compose(common_transforms)

    train_data_path = '/home/bincy/A2/CS-6910-A2/inaturalist_12K/train'
    test_data_path = '/home/bincy/A2/CS-6910-A2/inaturalist_12K/val'

   
    full_train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)

    # splitting 80:20 randomly
    validation_size = 0.2
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=42)
    train_indices, validation_indices = next(stratified_splitter.split(np.array(full_train_dataset.targets), np.array(full_train_dataset.targets)))

    # Create subsets for training and validation
    train_subset = Subset(full_train_dataset, train_indices)
    validation_dataset = datasets.ImageFolder(train_data_path, transform=test_transform)  
    validation_subset = Subset(validation_dataset, validation_indices)

   
    test_dataset = datasets.ImageFolder(test_data_path, transform=test_transform)

   
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, validationloader, testloader

class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, num_filters=[32, 32, 32, 32, 32],kernel_size=3, pool_size=2,drop_conv=0.2, drop_dense=0.3,dense_neurons=1000,activation="ReLU",activation_dense="ReLU",use_batch_norm=True):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        activations = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(), 
        "Mish": nn.Mish()}
        for out_channels in num_filters:
            kernel_size=kernel_size
            padding=kernel_size//2
            activation_function=activations.get(activation, nn.ReLU())

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    #nn.ReLU(),
                    activation_function,
                    nn.MaxPool2d(pool_size),
                    nn.Dropout(drop_conv)  
                )
            )
            
            if use_batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(out_channels))  
            in_channels = out_channels
          
            
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1] * (256 // (pool_size**len(num_filters))) * (256 // (pool_size**len(num_filters))), dense_neurons),
            nn.ReLU(),
            nn.Dropout(drop_dense),  
            nn.Linear(dense_neurons, num_classes)
        )
        
    def forward(self, x):
        
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.dense_layers(x)
        return x


# Instantiate the model
model = CNN()
print(model)
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


#def fit(config):
def fit(filter_mult=1,filter_num=32,drop_conv=0.1,drop_dense=0.2,use_batch_norm='True',batch_size=32,dense_neurons=1000,kernel_size=3,activation="ReLU",activation_dense="ReLU",epochs=10,use_data_augmentation=True,wandb_project="CS6910 A2"):
    
    wandb.login() 
    wandb.init(wandb_project,name="part a")
    
    loss_epoch_arr = []
    train_accuracy_arr=[]
    validation_accuracy_arr=[]
    loss_epoch_val=[]
    n=filter_num
    # early stop 
    patience = 5  # Number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')  # Initialize best validation loss
    best_val_acc = 0  # Optionally, you could also track best validation accuracy
    no_improvement_count = 0  # 
    
    trainloader,validationloader,testloader=prepare(batch_size,use_data_augmentation)
   
    device = torch.device("cuda:0")
    if filter_mult == 1:
        num_filters = [n, n, n, n, n]
    if filter_mult==2:
        num_filters = [n* (2 ** i) for i in range(5)]
    if filter_mult==0.5:
        num_filters = [n//(2 ** i) for i in range(5)]
    device = torch.device("cuda:0")
    net = CNN(num_filters=num_filters,drop_conv=drop_conv,drop_dense=drop_dense,use_batch_norm=use_batch_norm,dense_neurons=dense_neurons,kernel_size=kernel_size,activation=activation).to(device)
       
    print(net)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters())  
    n_iters = np.ceil(10000/batch_size)
  
    min_loss = 1000
    for epoch in range(epochs):

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            train_loss = loss_fn(outputs, labels)
            train_loss.backward()
            opt.step()
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            if min_loss > train_loss.item():

                min_loss = train_loss.item()
                best_model = copy.deepcopy(net.state_dict())
                print('Min loss %0.2f' % min_loss)
                
            if i % 1000 == 0:
                print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, train_loss.item()))
            
        loss_epoch_arr.append(train_loss.item())
        train_accuracy,t_loss=evaluation(trainloader,net)
        net.eval()
        validation_accuracy,validation_loss=evaluation(validationloader,net)
        train_accuracy_arr.append(train_accuracy)
        validation_accuracy_arr.append(validation_accuracy)
        loss_epoch_val.append(validation_loss)
            
        print('Epoch: %d/%d, validation acc: %0.2f,Train acc: %0.2f ,val loss: %0.2f, train loss: %0.2f'%(
            epoch, epochs, 
            validation_accuracy,train_accuracy,validation_loss,train_loss.item()))
        
        wandb.log({"accuracy_train": train_accuracy, "accuracy_validation": validation_accuracy, "loss_train": train_loss.item(), "loss_validation": validation_loss, 'epochs': epoch})
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_val_acc = validation_accuracy  # Update best validation accuracy if tracking
            best_model = copy.deepcopy(net.state_dict())  # Save the best model state
            no_improvement_count = 0  # Reset counter
            print('Improvement found at epoch {}: validation loss: {}, validation accuracy: {}'.format(epoch, validation_loss, validation_accuracy))
        else:
            no_improvement_count += 1
            print('No improvement in epoch {}. Current validation loss: {}, Best validation loss: {}'.format(epoch, validation_loss, best_val_loss))

        # Early stopping check
        if no_improvement_count >= patience:
            print('No improvement in validation loss for {} consecutive epochs. Stopping training...'.format(patience))
            break  # Exit the training loop
    
        
    
        net.train()
        
        
    plt.plot(loss_epoch_arr)
   
    plt.plot(range(1, epochs + 1), train_accuracy_arr, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), validation_accuracy_arr, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("saving the model--wait")
    net.eval()
    #best_model_accuracy,best_model_loss=evaluation(validationloader,net)
    path_to_save='/home/bincy/A2/CS-6910-A2/mymodel.pth'
    torch.save(net.state_dict(), path_to_save)
    #print('validation acc from best model: %0.2f,val loss: %0.2f'%(best_model_accuracy,best_model_loss))
    print("model saved")
    print("evaluating on test data")
    
    best_model_test_accuracy,best_model_test_loss=evaluation(testloader,net)
    true_labels, predictions = get_all_preds(net, testloader)  
    classes = testloader.dataset.classes 

    plot_confusion_matrix(true_labels, predictions, classes)
    print('test acc from best model: %0.2f,test loss: %0.2f'%(best_model_test_accuracy,best_model_test_loss))
    class_names = ('Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')
    show_random_predictions(testloader, net, class_names)
    print("the model uses early stopping to prevenet overfitting on train data with tolerence 5")
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity',default="bincyantonym")
    parser.add_argument('--wandb_project',default="CS6910 A2")
   
    parser.add_argument('--filter_mult', type=int,default=1,help='choices: ["1,2,0.5"]')
    parser.add_argument('--filter_num', type=int,help='choices: ["32 only since memory exceeded when 64 was used with multiplier"]',default=32)
    parser.add_argument('--kernel_size', type=int, default =3,
                    help='the kernel size 3,5,7,11')
    parser.add_argument('--drop_conv', type=float, default=0.2,help='drop out value to be used in conv layers.')
    parser.add_argument('--drop_dense', type=float, default=0.3, help='drop out for dense layers')
    parser.add_argument('--dense_neurons',type=int, default=1000, help='number of neurons in fully connected layers')
    parser.add_argument('--activation', default='relu',help='ReLU GELU SiLU Mish Tanh for conv layers')
    parser.add_argument('--activation_dense', default='relu',help='ReLU GELU SiLU Mish Tanh for dense layers')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size used to train neural network,64 and above resulted in cuda error.')
    
    parser.add_argument('--epochs', type=int, default=10,help='	Number of epochs to train neural network.')
    parser.add_argument('--use_data_augmentation', type=bool, default=True,help='data augmentation to be done or not.')
    parser.add_argument('--use_batch_norm', type=bool, default=True,help='batch normalization to be done or not.')
    
    args = parser.parse_args()
    fit(args.filter_mult,args.filter_num,args.drop_conv,args.drop_dense,args.use_batch_norm,args.batch_size,args.dense_neurons,args.kernel_size,args.activation,args.activation_dense,args.epochs,args.use_data_augmentation,
                  args.wandb_project)
    wandb.run.save()
    wandb.run.finish()