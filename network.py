import torch
import torch.nn as nn
import numpy as np


class VanillaNetwork(nn.Module):
    """ Implemention of a feed-forward neural network with ReLu activation. """

    def __init__(self, n_hidden, nodes, seed=0):
        """
        Initialize parameters and build network.

        Params
        =======
            n_hidden (int): Number of hidden layers
            nodes (List): Output dimension of each layer.
                          This includes the input layer and excludes the output layer.
            seed (int): Random seed
        """

        super(VanillaNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Sequential(nn.Linear(1, nodes[0]), nn.ReLU())
        self.lin_layers = []
        for i in range(n_hidden):
            self.lin_layers.append(nn.Sequential(nn.Linear(nodes[i], nodes[i+1]), nn.ReLU()))
        self.output_layer = nn.Linear(nodes[-1], 1)

        # initialize random weights
        torch.nn.init.xavier_uniform_(self.input_layer[0].weight)
        for layer in self.lin_layers:
            torch.nn.init.xavier_uniform_(layer[0].weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """
        Forward data through the network.
        """
        x = self.input_layer(x)
        for layer in self.lin_layers:
            x = layer(x)
        return self.output_layer(x)


def train_network(network, optimizer, n_iter, batch_size, f, noise, seed=0):
    """
    Initialize parameters and build network.

    Params
    =======
        network (torch.nn.Module): Neural network to train,
        optimizer (torch.nn.optim): Optimizer to train the neural network.
        n_iter (int): Number of training steps to perform.
        batch_size (int): Batch size to use.
        f (callable): real-valued function from R to R with support in [-1, 1] to approximate
        noise (callable): Noise perturbation of target y.
        seed (int): Random seed.
    """

    assert callable(f)
    assert callable(noise)

    np.random.seed(seed)

    network.train()
    for _ in range(n_iter):
        optimizer.zero_grad()
        # create training batch
        x = torch.from_numpy(np.random.uniform(low=-1, high=1, size=batch_size)).float()
        #x = torch.from_numpy(np.linspace(start=-1, stop=1, num=batch_size)).float()
        y = f(x)
        # perturb label y by noise
        y_noisy = torch.from_numpy(noise(size=batch_size)).float() + y
        # predict y from x
        y_hat = network(x.unsqueeze(1))
        # calculate prediction loss (MSE)
        loss = (y_hat - y_noisy).pow(2)
        # back propagate
        loss = loss.mean()
        loss.backward()
        # update the network weights
        optimizer.step()
    network.eval()