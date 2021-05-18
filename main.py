from network import VanillaNetwork, train_network
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up neural network
    n_hidden = 10
    nodes = [1] * (n_hidden + 1)
    network = VanillaNetwork(n_hidden=n_hidden, nodes=nodes).to(device)

    # set up optimizer (we use stochastic gradient descent)
    lr = 0.001
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    # define function to approximate
    def f(x):
        return 2*x.pow(2) - x

    # define noise perturbation of function values. here, we use white noise.
    def noise(size):
        return np.random.normal(loc=0.0, scale=0.01, size=size)

    # train neural network
    train_network(network=network, optimizer=optimizer, n_iter=int(1e5), batch_size=1, f=f, noise=noise)

    # plot approximated function
    num_samples = 100
    X = torch.from_numpy(np.linspace(start=-1, stop=1, num=num_samples)).float()
    y_noisy = f(X) + torch.from_numpy(noise(size=num_samples)).float()
    y_hat = network(X.unsqueeze(1))

    plt.scatter(X.detach().numpy(), y_noisy.detach().numpy(), s=5, label="True Data")
    plt.plot(X, y_hat.detach().numpy(), c="r", label="NN Approximation")
    plt.xlabel("x")
    plt.ylabel("$f(x)+\eta$")
    plt.show()





