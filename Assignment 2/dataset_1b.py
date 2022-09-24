# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:52:08 2021

@author: 91960
"""

# In[1]:
# Import Relevant Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score

import torch
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

#############################################################################

# In[2]:
# Read Data
# train data
train_data = pd.read_csv("datasets/Dataset_1a/train.csv",header=None)
    
# shuffle dataset
train_data = train_data.sample(frac=1)

# get train data
train_data = np.array(train_data)

# test data
test_data = pd.read_csv("datasets/Dataset_1a/dev.csv",header=None)

# shuffle dataset
test_data = test_data.sample(frac=1)

# get test data
test_data = np.array(test_data)

##############################################################################

# In[3]:
# Split training data
# length of data for fit
train_len = int(np.shape(train_data)[0])
val_len = int(np.shape(test_data)[0]*0.5)
test_len = int(np.shape(test_data)[0]*0.5)

X_train = train_data[:,0:2]
X_val = test_data[0:val_len,0:2]
X_test = test_data[val_len:val_len+test_len,0:2]

y_train = train_data[:,2]
y_val = test_data[0:val_len,2]
y_test = test_data[val_len:val_len+test_len,2]

##############################################################################

# In[4]:

# Build MLFFNN with 2 hidden layers using pytorch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # in_features = input dimension out_features -> to be tuned
        self.fc1 = nn.Linear(in_features=2, out_features=8)  
        
        # out_features -> to be tuned 
        self.fc2 = nn.Linear(in_features=8, out_features=16)
        
        # out_features = number of classes to be predicted
        self.fc3 = nn.Linear(in_features=16, out_features=4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# In[5]:
    
# Initialize neural network class
net = Net()

# # transfer the model to GPU if available
# if torch.cuda.is_available():
#     print("using GPU")
#     net = net.cuda()
    
# In[6]:

########################################################################
# Define a Loss function and optimizer

# Both variables are to be tuned
num_epochs = 20         # desired number of training epochs.
learning_rate = 0.01

# loss function and optimizers can be changed
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

num_params = np.sum([p.nelement() for p in net.parameters()])
print(num_params, ' parameters')

# In[7]:
# create dataloader for loading data

class myDataset(torch.utils.data.Dataset):
  
  #'Characterizes a dataset for PyTorch'
  def __init__(self,X,y,total_samples):
        #'Initialization'
        self.X = X
        self.y = y
        self.total_samples = total_samples
  
  def __len__(self):
        #'Denotes the total number of samples'
        return self.total_samples

  def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        # Load data and get label
        
        x_data = self.X[index,:]
        y_data = self.y[index]
        
        return x_data,y_data

# batch size can be changed to make epochs faster
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 0}

# training dataser
training_set = myDataset(X_train,y_train,train_len)

training_generator = torch.utils.data.DataLoader(training_set, **params)

# validation dataset
validation_set = myDataset(X_val,y_val,val_len)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# test dataset
test_set = myDataset(X_test,y_test,test_len)

params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 0}

test_generator = torch.utils.data.DataLoader(test_set, **params)

# In[8]:
    
def validation(model, loader):
    total_loss = 0
    accuracy = []
    tq = partial(tqdm, position=0, leave=True)
    
    model.eval()
    with torch.no_grad():
        for X, y in tq(loader):
          X = X.float()
          y = y.long()
          
          # if torch.cuda.is_available():
          #   X = X.cuda()
          #   y = y.cuda()
        
          prediction = model(X)
          
          loss = criterion(prediction, y)
          
          prediction = F.softmax(prediction)
          
          acc = np.mean(np.array((torch.argmax(prediction,1) == y)))*100
          
          
          total_loss += loss.item()
          accuracy.append(acc)

    # print('Validation Accuracy: ', np.mean(np.array(accuracy)))   
    return total_loss/len(loader),np.mean(np.array(accuracy))


# In[9]:
    
def test(model, loader):
    y_pred = []
    accuracy = []
    tq = partial(tqdm, position=0, leave=True)
    
    model.eval()
    with torch.no_grad():
        for X, y in tq(loader):
          X = X.float()
          y = y.long()
          
          if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        
          prediction = model(X)
          
          prediction = F.softmax(prediction)
          
          acc = np.mean(np.array((torch.argmax(prediction,1) == y)))*100

          accuracy.append(acc)

          prediction = torch.argmax(prediction,1)
          y_pred = y_pred + list(np.array(prediction))

    print('Test Accuracy: ', np.mean(np.array(accuracy)))   
    return y_pred,np.mean(np.array(accuracy))

# In[10]:
    
tq = partial(tqdm, position=0, leave=True)

print('Start Training')
train_loss_list = []
train_accuracy_list = []

validation_loss_list = []
validation_accuracy_list = []

for epoch in range(0,num_epochs):
  print('epoch ', epoch + 1)
  loss = 0
  train_accuracy = []
  for X,y in tq(training_generator):
    # zero the parameter 
    X = X.float()
    y = y.long()
    # y = torch.squeeze(y,1)
    
    # if torch.cuda.is_available():
    #     X = X.cuda()
    #     y = y.cuda()
    
    
    optimizer.zero_grad()
    
    output = net(X)
    loss = criterion(output,y)
    
    loss.backward()
    optimizer.step()

    loss += loss.item()
    
    prediction = F.softmax(output)
    
    accuracy = np.mean(np.array((torch.argmax(prediction,1) == y)))*100
    train_accuracy.append(accuracy) 

  
  train_loss_list.append([loss/len(training_generator)])
  # print(loss/len(training_generator))
  
  train_accuracy_list.append(np.mean(np.array(train_accuracy)))
  
  val_loss,val_accuracy = validation(net, validation_generator)

  validation_loss_list.append(val_loss)
  validation_accuracy_list.append(val_accuracy)
  
  
  
  
# In[11]:
    
plt.plot(train_loss_list,label='Training Loss')
plt.plot(validation_loss_list,label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.legend()
plt.plot()

# In[12]:
    
plt.plot(train_accuracy_list,label='Training Accuracy')
plt.plot(validation_accuracy_list,label='Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.legend()
plt.show()

# In[13]:
    
y_pred, test_accuracy = test(net,test_generator)

c_matrix  = confusion_matrix(y_test, y_pred)

afig, ax = plot_confusion_matrix(conf_mat=c_matrix,figsize=(7,7),cmap=plt.cm.RdBu)
ax.set(title = "Confusion Matrix")

# In[14]:
    
# Decision Region Plots #####################################################
# define bounds of the domain
min1, max1 = X_train[:, 0].min()-1, X_train[:, 0].max()+1
min2, max2 = X_train[:, 1].min()-1, X_train[:, 1].max()+1

# define the x and y scale
x1_grid = np.arange(min1, max1, 0.1)
x2_grid = np.arange(min2, max2, 0.1)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)

c1, c2 = x1_grid.flatten(), x1_grid.flatten()
c1, c2 = x1_grid.reshape((len(c1), 1)), x2_grid.reshape((len(c2), 1))

x = np.hstack((c1,c2))

tq = partial(tqdm, position=0, leave=True)
    
net.eval()
y_pred = []

with torch.no_grad():
    x = torch.from_numpy(x)
    x = x.float()
      
    if torch.cuda.is_available():
      x = x.cuda()
            
    prediction = net(x)
    prediction = F.softmax(prediction)
    prediction = torch.argmax(prediction,1)
    y_pred = y_pred + list(np.array(prediction.cpu()))
    
y_pred = np.array(y_pred)

x3_grid = y_pred.reshape(x1_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(x1_grid, x2_grid, x3_grid, cmap='Paired')
ax.scatter(X_train[:,0],X_train[:,1],marker='x')
ax.scatter(X_test[:,0],X_test[:,1],marker='x')
ax.set_xlabel('x1',fontsize=20)
ax.set_ylabel('x2',fontsize=20)
ax.set_title('MLFNN', fontsize=20)

########################################################################
# In[15]:
    
# Non-linear SVM
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# In[16]:
    
# Gaussian Kernel
clf = OneVsRestClassifier(SVC(kernel='rbf')).fit(X_train, y_train)
y_pred = clf.predict(X_val)

# In[17]:

# Gaussian Kernel
clf = OneVsRestClassifier(SVC(kernel='poly')).fit(X_train, y_train)
y_pred = clf.predict(X_val)  


# In[18]:













