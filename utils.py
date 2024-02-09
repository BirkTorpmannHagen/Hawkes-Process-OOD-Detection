import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_in_json(path_to_json):
    with open(path_to_json, 'r') as json_file:
            data = json.load(json_file)
    return data

def plot_heatmap(matrix, title, path_to_figures, nr):
    """
    Plots a heatmap of the given matrix.

    :param matrix: A 2D numpy array or a tensor.
    :param title: Title of the heatmap.
    """
    # If the input is a tensor, convert it to a numpy array
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.savefig(path_to_figures / f"alpha_heatmap_{nr}")


def predict_first_spike(output_spike):
    # Create a mask for neurons that never fire
    no_spike_mask = torch.sum(output_spike, dim=1) == 0

    # Find the first spike time for each neuron
    first_spike_times = torch.argmax(output_spike, dim=1)
    
    # Handle neurons that never fire
    # Set their first spike times to a large value
    first_spike_times[no_spike_mask] = output_spike.size(1)

    # If all neurons never fire, return a default value (e.g., -1)
    if no_spike_mask.all():
        return torch.tensor(-1)

    # Return the index (class) of the neuron that fired first
    return torch.argmin(first_spike_times)


def plot_spike_trains(spike_trains, path_to_fig, title="Spike Trains"):
    fig, ax = plt.subplots(figsize=(10, len(spike_trains)))

    for i, spike_train in enumerate(spike_trains):
        spikes = torch.nonzero(spike_train).flatten().numpy()
        ax.eventplot(spikes, orientation='horizontal', lineoffsets=i, colors='black', linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.set_yticks(range(len(spike_trains)))
    ax.set_yticklabels([f'Neuron {i+1}' for i in range(len(spike_trains))])
    plt.savefig(path_to_fig)

class ArgumentIterator:
    #add arguments to an iterator for use in parallell processing
    def __init__(self, iterable, variables):
        self.iterable = iterable
        self.index = 0
        self.variables = variables

    def __next__(self):
        if self.index >= len(self.iterable):
            raise StopIteration
        else:
            self.index += 1
            return self.iterable[self.index-1], self.variables

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)
