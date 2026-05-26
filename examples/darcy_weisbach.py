
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
Q = 1_000
N = 10_000 
n = 100
T_max = 1
x_max = 1

def friction(u, visc, rho, D):
    Re = rho*u*D/(visc)
    if Re < 2300:
        return 64/Re
    else:
        return 0.316*(Re**0.25)

    """
    return torch.where(Re < 2300,
                       64/Re,
                       0.316*(Re**0.25)
                      )
    """


# Network Parameter Settings

input_labels = ["x"]
output_labels = ["p_tilde"]
n_inputs = len(input_labels)
n_outputs = len(output_labels)
n_epochs = 50000 
lr = 1e-3 
x_intv = [0, 1]
s_intv = np.stack([[1e5], [1e6]], axis=0)
p_ref = 1e6

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
initial_state_test = np.array([1, 0, 0])

# Data definition for boundary condition
# For t>0, u=1 for x=0.

x_bc = np.zeros(3*n)[:, None] 
t_bc = np.linspace(1e-8, T_max, 3*n)[:, None]

data_bc = np.hstack([x_bc, t_bc])


# PDE
f = "D(p_tilde, x) + rho*g*sin(theta)/p_ref + friction(u, visc, rho, D)*rho*(u**2)/(2*D*p_ref)"

U_t = np.random.uniform(low=x_intv[0], high=x_intv[1], size=Q)
U_s = np.random.uniform(low=s_intv[0], high=s_intv[1], size=(N, 3))

branch_input_train = np.tile(U_s[:, None, :], (1, Q, 1)).reshape(N * Q, -1)
trunk_input_train = np.tile(U_t[:, None], (N, 1))

branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
trunk_input_test = np.sort(U_t[:, None], axis=0)

initial_states = U_s

input_labels = ["x"]
output_labels = ["p_tilde"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
lambda_2 = 0.0  # Penalty factor for the L² regularization
n_epochs = 400_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm


def model():

    import numpy as np

    from simulai.models import ImprovedDeepONet
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 100
    n_inputs_b = 3
    n_inputs_t = 1
    n_outputs = 1

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 6 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 6 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    # Instantiating and training the surrogate model
    trunk_net = ConvexDenseNetwork(**trunk_config)
    branch_net = ConvexDenseNetwork(**branch_config)

    encoder_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    net = ImprovedDeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        encoder_trunk=encoder_trunk,
        encoder_branch=encoder_branch,
        var_dim=n_outputs,
        rescale_factors=np.array([1]),
        devices="gpu",
        model_id="net",
    )

    return net


net = model()

# Residual

residual = SymbolicOperator(
    expressions=[f],
    input_vars=["x"],
    output_vars=["p_tilde"],
    inputs_key="input_trunk",
    external_functions={'friction': friction},
    constants = {'g': 9.81, 'D': 0.20, 'visc': 1e-6, 'theta': np.pi/2, 'rho':
                 850, 'u': 1, 'p_ref': p_ref},
    function=net,
    engine="torch",
    device="cpu",
)


# Maximum derivative magnitudes to be used as loss weights
penalties = [1e6]
batch_size = 10_000

optimizer_config = {"lr": lr}

input_data = {"input_branch": branch_input_train, "input_trunk": trunk_input_train}

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
    summary_writer=True,
)

params = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "residual": residual,
    "initial_input": {"input_trunk": np.zeros((N, 1)), "input_branch": initial_states},
    "initial_state": initial_states,
    "weights_residual": [1],
    "weights": [1],
}

optimizer.fit(
    op=net,
    input_data=input_data,
    n_epochs=n_epochs,
    loss="opirmse",
    params=params,
    device="gpu",
    batch_size=batch_size,
)

# Saving model
print("Saving model.")
saver = SPFile(compact=False)
saver.write(save_dir=save_path, name="rober_deeponet", model=net, template=model)

approximated_data = net.eval(
    trunk_data=trunk_input_test, branch_data=branch_input_test
)

for ii in range(n_outputs):
    plt.plot(approximated_data[:, ii], label="Approximated")
    plt.legend()
    plt.savefig(f"rober_deeponet_time_int_{ii}.png")
    plt.show()
