# CS-6910-A2

The repository includes:

1. part_a_train.py
2. part_b_train.py
3. inference.py
4. train_sweep.ipynb
5. pretrain.ipynb
6. dataload.ipynb
   

This repository has files to build and train a CNN from scratch and use a pretrained model.


Part A: build from scratch, save the model, test on the testloader 
1. use file part_a_train.py to build the model from scratch.
It can be run as **python part_a_train.py --arguments**

         The arguments are loaded using argparser as:
                   --wandb_entity',default="bincyantonym")
                   --wandb_project',default="CS6910 A2")
         
        --filter_mult', type=int,default=2,help='choices: ["1,2,0.5"]')
        --filter_num', type=int,help='choices: ["32 only since memory exceeded when 64 was used with multiplier"]',default=32)
          --kernel_size', type=int, default =3,
                          help='the kernel size 3,5,7,11'
          --drop_conv', type=float, default=0.2,help='drop out value to be used in conv layers.'
          --drop_dense', type=float, default=0.3, help='drop out for dense layers')
         --dense_neurons',type=int, default=500, help='number of neurons in fully connected layers'
          --activation', default='relu',help='ReLU GELU SiLU Mish Tanh for conv layers'
         --activation_dense', default='relu',help='ReLU GELU SiLU Mish Tanh for dense layers'
         --batch_size',type=int,default=32,help='Batch size used to train neural network,64 and above resulted in cuda error.'
          --epochs', type=int, default=10,help='	Number of epochs to train neural network.'
      --use_data_augmentation', type=bool, default=True,help='data augmentation to be done or not.'
          --use_batch_norm', type=bool, default=True,help='batch normalization to be done or not.'
          
 ** The file does complete training with early stopping and saves the model which can be loaded using state_dict.**

   2. train_part_b.py contain train file which takes a pretrained model from resnet and uses it to train the inaturalist dataset.
     
   The confusion matrix is plotted nad accuracy printed. It can be tuned by changing optimizer, learning rate, batch size and drop out.
   To run the file use command **python train_part_b.py --arguments
   The arguments used are --batch_size,de

  3. The inference.py file can be used to load the model that is saved after training part a (optionally done ), load the test data, evaluate and plot confusion matrix and a grid of predicted label of images.[ when traing if you are changing dense neurons make sure to change the netork config in CNN class.Its default is 1000]
      run it as **python inference.py** [ load cnn built by us with 1000 dense neurons, in case you are changing model, change the CNN definition of dense neurons ]
 
     
  5. train_sweep.ipynb is the file that contains the complete dataload, training with early stop and configuurations for sweep
  6. pretrain.ipynb is the file that loads the resnet model from torchvision and do train
  7. dataload.ipynb file shows detailed view of data and shows filter output of first convolution layer.

  
  

  





