
import sympy as sp
import numpy as np
import torch
import matplotlib.pyplot as plt
from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator

# Imports to neural network
from simulai.models import ImprovedDenseNetwork
from simulai.regression import SLFNN, ConvexDenseNetwork, DenseNetwork
from simulai.tokens import D
from simulai.file import SPFile


# Imports for visualization and animation
from matplotlib import animation, rc
from IPython.display import HTML 

# Setting seed to aid reproducibility
np.random.seed(1234)


# Solution parameters and dataset building

nx = 401  # Number of points to be plotted
dx = 1./(nx-1) # Distance between each plotted point
nt = 301  # Number of points in which the time will be divided (snapshots)
dt = 1./(nt-1)  # Time between snapshots


# Parameter Setting

N = 10_000 
n = 100
T_max = 1
x_max = 1

def friction(u, visc, rho, D):
    Re = rho*u*D/(visc)
    return torch.where(Re < 2300,
                       64/Re,
                       0.316*(Re**0.25)
                      )



# Network Parameter Settings

input_labels = ["x", "t"]
output_labels = ["u"]
n_inputs = len(input_labels)
n_outputs = len(output_labels)
n_epochs = 50000 
lr = 1e-3 

boundary_penalties = [1]
weights_residual =  [1]
initial_penalty = 1

# PDE training and test set definition
x_eval = (np.random.rand(N) * x_max)[:, None]
time_eval = (np.random.rand(N) * T_max)[:, None]

# Regular grid
X_DIM = 100
T_DIM = 100

dx = (x_max) / X_DIM
dt_ = (T_max) / T_DIM

grid = np.mgrid[dt_ : T_max + dt_ : dt_, 0:x_max:dx]

X_train = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])


# In[ ]:


# Dataset Definition for Initial Condition
# u=0 at t=0 for any x

x_init = np.linspace(0, x_max, 3*n)[:, None]
t_init = np.zeros(3*n)[:, None]

data_initial = np.hstack([x_init, t_init])

u_initial = np.zeros(3*n)[:, None]

# Data definition for boundary condition
# For t>0, u=1 for x=0.

x_bc = np.zeros(3*n)[:, None] 
t_bc = np.linspace(1e-8, T_max, 3*n)[:, None]

data_bc = np.hstack([x_bc, t_bc])


# In[ ]:


# Setting the function that will build the training net
# It is possible to choose two different types of networks

def model(net_type="Dense"):

  # If it is a fully connected network

  if net_type=="Dense":

    # Configuration for the fully-connected network
    config = {
        "layers_units": 6*[128],
        "activations": "tanh",
        "input_size": 2,
        "output_size": 1,
        "name": "net",
    }

    # Instantiating and training the surrogate model
    net = DenseNetwork(**config)

    return net

  # If you want to use the ImprovedDenseNetwork
  elif(net_type=="Improved"):

    # Configuration for the ImprovedDenseNetwork
    config = {
        "layers_units": 6*[128],
        "activations": "elu",
        "input_size": 2,
        "output_size": 1,
        "name": "net",
    }

    # Instantiating and training the surrogate model
    densenet = ConvexDenseNetwork(**config)
    encoder_u = SLFNN(input_size=n_inputs, output_size=128, activation="tanh")
    encoder_v = SLFNN(input_size=n_inputs, output_size=128, activation="tanh")

    net = ImprovedDenseNetwork(
        network=densenet,
        encoder_u=encoder_u,
        encoder_v=encoder_v,
        devices="gpu",
    )

    return net


# In[ ]:


# PDE

f = "D(p, x) + rho*g*sin(theta) + friction(u, visc, rho, D)*rho*(u**2)/(2*D)"

# Network
net = model("Dense")

# Residual

residual = SymbolicOperator(
    expressions=[f],
    input_vars=["x"],
    output_vars=["p"],
    external_functions={'friction': friction},
    constants = {'g': 9.81, 'D': 0.20, 'visc': 1e-6, 'theta': np.pi/2, 'rho':
                 850, 'u': 1},
    function=net,
    engine="torch",
    device="cpu",
)


