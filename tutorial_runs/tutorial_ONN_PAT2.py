# %% 

# Load Packages
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

#defining an effective J matrix
def C2Q(C):
    Q = (C + C.T)/2
    C_colsum = -torch.sum(Q, axis=0)
    inds = list(range(C.shape[0]))
    Q[inds, inds] = C_colsum
    return Q

#Define the ODE for the model with this Qmatrix
#Note this ODE can be used as a model for both the small and large oscillator network
def ode_model(z, C, e):
    Q = C2Q(C)
    q = z[..., 0]
    p = z[..., 1]
    output = torch.empty_like(z)
    output[..., 0] = p
    sin_q = torch.sin(q)
    output[..., 1] = -sin_q + e + sin_q@Q
    return output

#parameters associated with noise
J_noise = 0.2
η = 0.1
input_noise = 0.02
dim1 = 14*14
dim2 = 14*14 + 10

#Since the small and large networks require different noise coupling matrix, two separate
#ODE needs to be defined.

torch.manual_seed(0)
# Define the noise coupling matrice for the small network. This is fixed during training and testing. 
Q_noise_small = C2Q(J_noise*torch.randn(dim1, dim1))

#Define the "experiment" ODE for the small oscillator network
def ode_exp_small(z, C, e):
    Q = C2Q(C)
    q = z[..., 0]
    p = z[..., 1]
    output = torch.empty_like(z)
    output[..., 0] = p
    sin_q = torch.sin((1+η)*q)
    output[..., 1] = -sin_q + e + sin_q@Q + sin_q@Q_noise_small
    return output

# Define the noise coupling matrice for the large network. This is fixed during training and testing. 
Q_noise_large = C2Q(J_noise*torch.randn(dim2, dim2))

#Define the "experiment" ODE for the large oscillator network
def ode_exp_large(z, C, e):
    Q = C2Q(C)
    q = z[..., 0]
    p = z[..., 1]
    output = torch.empty_like(z)
    output[..., 0] = p
    sin_q = torch.sin((1+η)*q)
    output[..., 1] = -sin_q + e + sin_q@Q + sin_q@Q_noise_large
    return output

#import functions from utils.py for solving ODEs
from ode_utils import make_ode_map
from pat import make_pat_func

# make_ode_map is a function that returns the ode_map function, which takes
# in the initial condition x and returns the final state of the ODE after 
# evolving by time Nt*dt. 

#parameters associated with the ODE solver
Nt = 5 #checked that 5 points converges - (same classification accuracy is achieved)
t_end = 0.5 #how long the ODE is time-evolved for
dt = t_end/Nt #discretization step for the ODE solver

#The model for the parameterized input output map of both oscillator network 
#Here the function arguments of f is f(x, C, e), 
#where x is the input data, C is the coupling matrix of the network (as used in ode),
#and e is the bias
f_model = make_ode_map(ode_model, Nt, dt)
f_small_exp = make_ode_map(ode_exp_small, Nt, dt)
f_large_exp = make_ode_map(ode_exp_large, Nt, dt)
f_pat_small = make_pat_func(f_small_exp, f_model)
f_pat_large = make_pat_func(f_large_exp, f_model)

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Load Training & Test Data
num_workers = 0 # number of subprocesses to use for data loading
batch_size = 20 # How many samples per batch to load
valid_size = 0.2 # percentage of training set to use as validation


transform = transforms.Compose([
    transforms.Resize((14, 14)),  # Downsample by a factor of 2
    transforms.ToTensor()
])

# Choose the training and testing datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# %% 
# Visualization of Training Data
import matplotlib.pyplot as plt
#matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(images[idx]), cmap='gray')
#     # print out the correct label for each image
#     # .item() gets the value contained in a Tensor
#     ax.set_title(str(labels[idx].item()))

img = np.squeeze(images[1])
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
# %%
import torch.nn as nn
import torch.nn.functional as F
dim1 = 14*14
dim2 = 14*14 + 10
# define NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        super().__init__()
        self.fc1 = nn.Linear(dim1, dim1)
        self.fc2 = nn.Linear(dim2, dim2)
        
        self.output_fac = nn.Parameter(torch.tensor(1.0).float())
            
    # def forward(self, x):
        
    #     x = x.view(-1,14*14)
    #     #print(x)

    #     #Augmenting the data vector to include the velocity vector, i.e. dot q = 0
    #     x = torch.stack([x, torch.zeros_like(x)], dim=2)
    #     x = f(x, *self.fc1.parameters())
        
    #     #Get the positions of the pendulums
    #     x = x[..., 0]
    #     #print(x)
    #     #Augmenting the data vector to include the velocity vector, i.e. dot q = 0
    #     x = torch.stack([x, torch.zeros_like(x)], dim=2)
    #     #Augmenting the initial conditions, which are zeros, of the "class oscillators"
    #     x = torch.cat([x, torch.zeros([x.shape[0], dim2 - dim1, 2], device=x.device)], dim=1)
    #     #print(x)
    #     #Use the second oscillator network  
    #     x = f(x, *self.fc2.parameters())
        
    #     #return the positions of the "class oscillators"
    #     x = self.output_fac*(x[:, -10:, 0])

    #     return x
    
    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, 14*14)
        #print(x.shape)
        x = torch.stack([x, torch.zeros_like(x)], dim=2)

        # Use the small PAT function for the first transformation
        x = f_pat_small(x, self.fc1.weight, self.fc1.bias)

        x = x[..., 0]

        x = torch.stack([x, torch.zeros_like(x)], dim=2)
        x = torch.cat([x, torch.zeros([x.shape[0], 10, 2], device=x.device)], dim=1)

        # Use the large PAT function for the second transformation
        x = f_pat_large(x, self.fc2.weight, self.fc2.bias)
        #print(self.output_fac * (x[:, -10:, 0]))

        return self.output_fac * (x[:, -10:, 0])


    # def enforce_weights_constraints(self):

    #     # Makes matrices are 0 - 1 values 
    #     self.W1.data.clamp_(0, 1)
    #     self.W2.data.clamp_(0, 1)
    #     self.W3.data.clamp_(0, 1)

    # def custom_parameter_update(self):
    #     # Directly manipulate parameters as needed
    #     # Example: Scale W1 by a factor
    #     self.W1.data *= 0.9
    
# initialize the NN
model = Net()
print(model)

def check_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f"Gradient norm for {name}: {parameter.grad.norm().item()}")
        else:
            print(f"No gradient for {name}")

# %%
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.01
#optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
# # %%
# # number of epochs to train the model
n_epochs = 20
# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity
for epoch in range(n_epochs):
    # monitor losses
    train_loss = 0
    #valid_loss = 0

    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for data, label in train_loader:
        optimizer.zero_grad()    # clear the gradients of all optimized variables
        output = model(data)    # forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(output,label)    # calculate the loss
        loss.backward()    # backward pass: compute gradient of the loss with respect to model parameters
  
        #check_gradients(model)
        optimizer.step()    # perform a single optimization step (parameter update)
        #model.enforce_weights_constraints()    # Enforce weight constraints
        train_loss += loss.item() * data.size(0)    # update running training loss
        
     ######################    
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    for data,label in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output,label)
        # update running validation loss 
        valid_loss = loss.item() * data.size(0)
    
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_ONN_PAT.pt')
        valid_loss_min = valid_loss

# %%
model.load_state_dict(torch.load('model_ONN_PAT.pt'))
# %%
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
model.eval() # prep model for evaluation
for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
# %%
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()
# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))
# %%
