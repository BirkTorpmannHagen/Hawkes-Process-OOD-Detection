from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet_snn import SResnet
from tqdm import tqdm
from snn_resnet_training import train
from threading import Thread
from fit_hawkes_snn import *
from utils import ArgumentIterator


def parallel_process(data_index, model, spikes_per_layer, result_dict):
    print("working on", data_index)
    spikes = spikes_per_layer[data_index, :, :, :]
    spike_trains = convert_spike_trains(spikes.reshape(-1, spikes.size()[-1]).cpu())
    model.fit(spike_trains)
    mu = np.reshape(model.baseline, (spikes_per_layer.size()[1], spikes_per_layer.size()[2]))
    result_dict[data_index] = mu  # Store result in dictionary
    # print(mu)

def generate_matrices(dataset, num_blocks=1, n_filters=8, num_steps=10, img_size=32, num_cls=10):

    #load snn and perform hawkes process, generating a matrix for the last layer
    snn = SResnet(n=num_blocks, nFilters=n_filters, num_steps=num_steps, img_size=img_size, num_cls=num_cls)
    snn.cuda()  # Move the model to GPU
    snn.load_state_dict(torch.load(f"checkpoints/{dataset.__class__.__name__}/resnet.pt"))
    snn.eval()
    matrices = []
    for i, (x,y)  in enumerate(DataLoader(dataset, shuffle=True, batch_size=32)):
        if i>100:
            break
        x = x.cuda()
        _, spike_trains = snn(x)
        convolutional_layer_spikes = concate_neuron_st(spike_trains, neuron_dim_reshape)
        for layer, spikes_per_layer in convolutional_layer_spikes.items():
            if layer!=max(convolutional_layer_spikes.keys()):
                continue
            model = HawkesADM4(decay=1.0)  # You can adjust parameters as needed
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            threads = []
            results = {}

            for data_index in range(spikes_per_layer.size()[0]):
                thread = Thread(target=parallel_process, args=(data_index, model, spikes_per_layer, results))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            for data_index, mu in results.items():
                matrices.append(mu)

    merged = np.vstack(matrices)
    return matrices





if __name__ == '__main__':
    mnist_train = datasets.MNIST(root="../../Datasets/MNIST", train=True, transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]), download=True)
    mnist_val = datasets.MNIST(root="../../Datasets/MNIST", train=False, transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]), download=True)

    emnist = datasets.EMNIST(root="../../Datasets/emnist", split="letters", train=True, transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]), download=True)
    mnist_matrices = generate_matrices(mnist_train)
    mnist_val_matrices = generate_matrices(mnist_val)
    emnist_matrices = generate_matrices(emnist)
    np.save("mnist_matrices.npy", mnist_matrices)
    np.save("mnist_val_matrices.npy", mnist_val_matrices)
    np.save("emnist_matrices.npy", emnist_matrices)

