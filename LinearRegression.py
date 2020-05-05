_________________________________________________________________________________________________________________________________________
Author : Srimanth Tenneti
Topic : Linear Regression using Pytorch
_________________________________________________________________________________________________________________________________________

########################################################### Code ########################################################################

# Importing the libs
import torch
import torch.nn as nn  # Neural network constructor
import numpy as np # For data manipulation
import matplotlib.pyplot as plt # For plotting

# Creating data for classification
X = torch.randn(100 , 1)*10     # Creating the data using torch.randn(rows , columns)
y = X + 3*torch.randn(100 , 1)
plt.plot(X.numpy() , y.numpy(),'o') # For matplotlib to plot the inouts cannot be tensors so we have to convert them into numpy arrays
plt.xlabel('X') # Setting the X label
plt.ylabel('Y') # Setting the Y label

# Creating the model
class LR(nn.Module):
  def __init__(self , input_size , output_size):
    super(LR,self).__init__()
    self.linear = nn.Linear(input_size,output_size)
  def forward(self,x):
    pred = self.linear(x)
    return pred
    
# Creating the model instance
torch.manual_seed(1)  # Just for you to get similar output as mine
model = LR(1 , 1)
print(model)

# Collecting the weights and biases
[w ,b] = model.parameters()
def get_params():
  return(w[0][0].item(),b[0].item())
  
# Plotting function
def plot_fit(title):
  plt.title = title
  w1 , b1 = get_params()
  x1 = np.array([-30 , 30])
  y1 = w1*x1 +b1
  plt.plot(x1,y1,'r')
  plt.scatter(X,y)
  plt.show()
  
plot_fit("Linear") # Visualization of the line with random weights and bias.

#Loss anad Optimizer defenitions
criterion = nn.MSELoss() # MSE stands for Mean Square Error
optimizer = torch.optim.SGD(model.parameters() , lr = 0.0001) # SGD stands for Stochastic Gradient Decent , lr = learning rate

#Training
epochs = 100
losses = []

for i_epoch in range(epochs):
  y_pred = model.forward(X) # Forward pass
  loss = criterion(y_pred,y) # Finding the loss
  print("{}/{} Epoch,Loss {}".format(i_epoch,epochs,loss.item())) # Monitoring the loss
  losses.append(loss.item()) # Stroing the loss for later visualization
  optimizer.zero_grad()  #Setting the gradients to zero
  loss.backward()  # Backward pass
  optimizer.step() # Updates the parameters
  
# Loss vs epochs
plt.plot(range(epochs) , losses)
plt.xlabel('epochs --->')
plt.ylabel('Loss --->')

plot_fit("Trained") # Visualizing the trained network

  
  
