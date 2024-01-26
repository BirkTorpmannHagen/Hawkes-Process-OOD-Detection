import cma
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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


def plot_spike_trains(spike_trains, fig_name, title="Spike Trains"):
    fig, ax = plt.subplots(figsize=(10, len(spike_trains)))

    for i, spike_train in enumerate(spike_trains):
        spikes = torch.nonzero(spike_train).flatten().numpy()
        ax.eventplot(spikes, orientation='horizontal', lineoffsets=i, colors='black', linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.set_yticks(range(len(spike_trains)))
    ax.set_yticklabels([f'Neuron {i+1}' for i in range(len(spike_trains))])
    plt.savefig(Path(__file__).parent / "figures"/ f"{fig_name}.png")


class LIF(nn.Module):
    def __init__(self, time, weight_size, leakage, threshold):
        super(LIF, self).__init__()
        self.resting_potential = 0
        self.resting_time = 10
        self.membrane_potential = nn.Parameter(torch.zeros(weight_size, time, dtype=torch.float32), requires_grad=False)
        self.spikes = torch.zeros(weight_size, time, dtype=torch.float32)
        self.leakage_factor = leakage
        self.threshold = threshold
    
    def forward(self, input_signal, weights):
        blocker = torch.zeros_like(self.membrane_potential)
        self.spikes.zero_()
        for i in range(0,input_signal.size(1)):
            if i == 0:
                self.membrane_potential[:,i] = torch.matmul(input_signal[:, i, None].T, weights).squeeze() - self.leakage_factor
            else:
                self.membrane_potential[:,i] = self.membrane_potential[:, i - 1] + torch.matmul(input_signal[:, i, None].T, weights).squeeze() - self.leakage_factor
                blocker -= 1
                # Vectorized conditions
                spike_condition = self.membrane_potential[:, i] >= self.threshold
                reset_condition = self.membrane_potential[:, i] <= self.resting_potential

            # Set values based on conditions
                self.spikes[spike_condition,i] = 1
                self.membrane_potential[spike_condition,i] = 0
                self.membrane_potential[reset_condition,i] = self.resting_potential
                blocker[spike_condition] = self.resting_time
        
        return self.spikes
    
class NeuralNetwork(nn.Module):
    def __init__(self, time, hidden_size, output_size, leakage, threshold):
        super(NeuralNetwork, self).__init__()
        self.layer1 = LIF(time, hidden_size, leakage, threshold)
        self.layer2 = LIF(time, output_size, leakage, threshold)

    def forward(self, input_signal, weights):
        spikes1 = self.layer1(input_signal, weights=weights[0])
        return self.layer2(spikes1, weights=weights[1])

def fitness(weights, training_data, neural_net):
    # Convert the flattened weight array back to the required shape
    weight_1 = torch.tensor(weights[:input_size * hidden_size].reshape(input_size, hidden_size), dtype=torch.float32)
    weight_2 = torch.tensor(weights[input_size * hidden_size:].reshape(hidden_size, output_size), dtype=torch.float32)
    weights = [weight_1, weight_2]
    # Your code to evaluate the network and return the loss
    
    loss = 0
    # Iterate over training data, calculate and accumulate loss
    for input_train, expected_output in training_data:
        output_spike = neural_net(input_train, weights)
        predicted_label = torch.argmax(torch.sum(output_spike, dim=1))
        loss += nn.functional.binary_cross_entropy(predicted_label.float(), expected_output)  # Implement this
    return loss

if __name__ == "__main__":
    time = 100
    input_size = 1
    hidden_size = 10
    output_size = 2  # For binary classification
    leakage = 0.25
    threshold = 1

    # Generate Data
    t = torch.linspace(0, 1, time)
    sine_curve = 0.5 * (1 + torch.sin(10 * 3.1416 * t))
    cos_curve = 0.5 * (1 + torch.cos(15 * 3.1416 * t))
    train_sine_spike_trains = [(torch.bernoulli(sine_curve).unsqueeze(0),torch.tensor(1).float()) for _ in range(50)]
    train_rndm_spike_trains = [(torch.bernoulli(cos_curve).unsqueeze(0),torch.tensor(0).float()) for _ in range(50)]
    #train_rndm_spike_trains = [(torch.empty(1, time).uniform_(0, 1).round(),torch.tensor(0).float()) for _ in range(50)]
    training = train_sine_spike_trains + train_rndm_spike_trains

    test_sine_spike_trains = [(torch.bernoulli(sine_curve).unsqueeze(0),torch.tensor(1).float()) for _ in range(15)]
    test_rndm_spike_trains = [(torch.empty(1, time).uniform_(0, 1).round(),torch.tensor(0).float()) for _ in range(15)]
    testing = test_sine_spike_trains + test_rndm_spike_trains

    # Create Neural Network
    neural_net = NeuralNetwork(time, hidden_size, output_size, leakage, threshold)
    weights = [torch.empty(input_size, hidden_size).uniform_(0, 1),
               torch.empty(hidden_size, output_size).uniform_(0, 1)]

    # EA Setup
    options = {'maxiter': 15, 'tolfun': 5e-4}  # Adjust these parameters as needed
    initial_mean = np.random.rand((input_size * hidden_size) + (hidden_size * output_size))
    initial_std = 0.5  # Adjust this parameter as needed

    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)

    while not es.stop():
        solutions = es.ask()
        fitness_values = [fitness(s, training, neural_net) for s in solutions]
        fitness_values = [value.item() for value in fitness_values]
        es.tell(solutions, fitness_values)
        es.logger.add()  # optional, for logging
        es.disp()
    # Extract the best solution
    best_weights = es.result.xbest
    size_first_matrix = input_size * hidden_size

    # Split and reshape the weights
    best_weights_1 = best_weights[:size_first_matrix].reshape(input_size, hidden_size)
    best_weights_2 = best_weights[size_first_matrix:].reshape(hidden_size, output_size)

    # Convert to tensors
    best_weights_tensor = [torch.tensor(best_weights_1, dtype=torch.float32), 
                        torch.tensor(best_weights_2, dtype=torch.float32)]
    
    predictions = []
    labels = []
    input_trains = []
    hidden_trains = []
    output_trains = []
    correct_predictions = []
    labels = []
    for test_input, label  in testing:
        output_spike = neural_net(test_input, best_weights_tensor)
        predicted_label = torch.argmax(torch.sum(output_spike, dim=1))
        correct_predictions.append((predicted_label == label).item())

        predictions.append(predicted_label)
        labels.append(label)
        input_trains.append(test_input.tolist())
        hidden_trains.append(neural_net.layer1.spikes.tolist())
        output_trains.append(neural_net.layer2.spikes.tolist())

    predictions = np.array(predictions)
    labels = np.array(labels)
    spike_trains = {"input_trains": input_trains, 
                    "hidden_trains": hidden_trains,
                    "output_trains": output_trains, 
                    "correct": labels.tolist()}
    
    metrics = {"ACC": accuracy_score(predictions, labels), 
               "F1": f1_score(predictions, labels), 
               "Recall": recall_score(predictions, labels),
               "Precision": precision_score(predictions, labels)}
    results = {"data": spike_trains, "metrics": metrics}
    print(metrics)
    with open(Path(__file__).parent / "spike_trains"/ 'results_1.json', 'w') as f:
        json.dump(results, f)
    print("FILE SAVED")
    
