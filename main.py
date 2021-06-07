from network import VanillaNetwork, train_network
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up neural network
    n_hidden = 2
    nodes = [32] * (n_hidden + 1)
    network = VanillaNetwork(n_hidden=n_hidden, nodes=nodes).to(device)

    # set up optimizer (we use stochastic gradient descent)
    lr = 0.001
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    # define function to approximate
    def f(x):
        return 2*x.pow(2) - x

    # define noise perturbation of function values. Here, we use white noise.
    def noise(size):
        return np.random.normal(loc=0.0, scale=0.01, size=size)

    # train neural network
    train_network(network=network, optimizer=optimizer, n_iter=int(1e5), batch_size=16, f=f, noise=noise)

    # plot approximated function
    num_samples = 1000
    X = torch.from_numpy(np.linspace(start=-1, stop=1, num=num_samples)).float()
    y = f(X)
    y_hat = network(X.unsqueeze(1))

    plt.plot(X.detach().numpy(), y.detach().numpy(), label="True Data")
    plt.plot(X, y_hat.detach().numpy(), c="r", label="NN Approximation")
    plt.xlabel("x")
    plt.ylabel("$f(x)+\eta$")
    plt.show()





