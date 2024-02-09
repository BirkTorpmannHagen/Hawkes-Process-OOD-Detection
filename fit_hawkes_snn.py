from pathlib import Path
import os
import numpy as np
import torch 
from tick.hawkes import HawkesADM4
import matplotlib.pyplot as plt
import seaborn as sns

def store_baseline_array(array_to_save, path_to_array, name):
    with open(path_to_array / name, 'wb') as f:
        np.save(f, array_to_save)

def plot_heatmap(matrix, label, batch_nr, layer, file_nr, path):
    """
    Plots a heatmap of the given matrix.

    :param matrix: A 2D numpy array or a tensor.
    :param title: Title of the heatmap.
    """
    # If the input is a tensor, convert it to a numpy array
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.savefig(path / f"FileNr: {file_nr} BatchNr:{batch_nr} Layer:{layer} Class:{label}.png")
    plt.show()

def neuron_dim_reshape(spikes, batchsize, neurons, image_x, image_y, length):
    return spikes.reshape(batchsize, image_x, image_y, neurons * length)

def spatial_dim_reshape(spikes, batchsize, neurons, image_x, image_y, length):
    return spikes.reshape(batchsize, neurons, image_x * image_y, length)

def concate_neuron_st(conv_layer_st, reshape_func):
    reshaped_conv_layers = {}
    for layer_nr, spikes in conv_layer_st.items():
        if layer_nr != "labels":
            batchsize, neurons, image_x, image_y, length = spikes.size() #[batchsize, neurons, image_x, image_y, length]
            reshaped_conv_layers[layer_nr] = reshape_func(spikes, batchsize, neurons, image_x, image_y, length) #.reshape(batchsize, image_x, image_y, neurons * length)
        else:
            reshaped_conv_layers["labels"] = spikes
    return reshaped_conv_layers

def convert_spike_trains(spike_trains):
    converted_data = []
    for i in range(spike_trains.shape[0]):  # Loop over each spike train
        train = spike_trains[i]
        timestamps = torch.nonzero(train).numpy().flatten()  # Get indices of spikes
        timestamps = timestamps.astype(np.float64) 
        converted_data.append(timestamps)
    return converted_data

def load_in_spikes(path_to_folder, reshape_func, path_to_figures, path_to_store_data):
    data_points = []
    for file_nr, file in enumerate(os.listdir(path_to_folder)[1:]): #convolutional spike trains
        convolutional_layer_spikes = concate_neuron_st(torch.load(path_to_folder / file), reshape_func)
        for layer, spikes_per_layer in convolutional_layer_spikes.items():
            print(layer)
            model = HawkesADM4(decay = 1.0)  # You can adjust parameters as needed
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for data_index in range(0,spikes_per_layer.size()[0]):
                spikes = spikes_per_layer[data_index, :, :, :]
                spike_trains = convert_spike_trains(spikes.reshape(-1, spikes.size()[-1]).cpu())
                model.fit(spike_trains)
                mu = np.reshape(model.baseline, (spikes_per_layer.size()[1], spikes_per_layer.size()[2]))
                np.save(open(""))
                #store_baseline_array(mu, path_to_store_data, f"FileNr: {file_nr} BatchNr:{data_index} Layer:{layer} Class:{labels[data_index]}.npy")
                #plot_heatmap(mu, labels[data_index], data_index, layer, file_nr, path_to_figures)
        data_points.append(convolutional_layer_spikes)
    return data_points

def main():
    path_to_folder = Path(__file__).parent / "spike_trains" / "convolutional_spike_trains"
    path_to_store_data = Path(__file__).parent / "spike_trains" / "baseline_arrays"
    path_to_figures = Path(__file__).parent / "figures" / "baseline_heatmaps"
    dat = load_in_spikes(path_to_folder, neuron_dim_reshape, path_to_figures, path_to_store_data)
    
if __name__ == "__main__":
    main()