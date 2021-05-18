import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random 

from models import getSimpleNeuralNet

class Recombiner():
    """
    Enables recombination of 2 parents into two children
    """

    def __init__(self, layer_names, n_inputs, n_outputs):
        self.layer_names = layer_names
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def __call__(self, parent1, parent2):
        layer_weights_1, layer_weights_2, layer_biases_1, layer_biases_2 = \
            self.recombine(parent1, parent2)
        child1 = self.create_child(layer_weights_1, layer_biases_1)
        child2 = self.create_child(layer_weights_2, layer_biases_2)
        return child1, child2

    def recombine(self, parent1, parent2):
        # storage for child networks
        layer_weights_1 = []
        layer_biases_1 = []
        layer_weights_2 = []
        layer_biases_2 = []
        # Get dict of networks
        parent1dict = dict(parent1.named_children())
        parent2dict = dict(parent2.named_children())
        # check size of hidden layers
        h_units_1 = parent1dict[self.layer_names[0]]\
            .weight.detach().numpy().shape[0]
        h_units_2 = parent2dict[self.layer_names[0]]\
            .weight.detach().numpy().shape[0]
        # determine cutoff point
        cutoff = random.randint(1, min(h_units_1, h_units_2, self.n_inputs, self.n_outputs))
        # loop over relevant layers
        for layer in self.layer_names:
            # get parent weights and biases
            W1 = parent1dict[layer].weight.detach().numpy()
            W2 = parent2dict[layer].weight.detach().numpy()
            b1 = parent1dict[layer].bias.detach().numpy()
            b2 = parent2dict[layer].bias.detach().numpy()
            # determine parts to keep
            Keep1 = W1[:cutoff, :cutoff]
            Keep2 = W2[:cutoff, :cutoff]
            # determine parts to swap
            SwapOutputs1 = W1[:cutoff, cutoff:]
            SwapOutputs2 = W2[:cutoff, cutoff:]
            SwapInputs1 = W1[cutoff:, :]
            SwapInputs2 = W2[cutoff:, :]
            # recombination of weights
            WChild1 = np.concatenate((Keep1, SwapOutputs2), 1)
            WChild2 = np.concatenate((Keep2, SwapOutputs1), 1)
            WChild1 = np.concatenate((WChild1, SwapInputs2), 0)
            WChild2 = np.concatenate((WChild2, SwapInputs1), 0)
            # recombination of biases
            bChild1 = np.concatenate((b1[:cutoff], b2[cutoff:]), 0)
            bChild2 = np.concatenate((b2[:cutoff], b1[cutoff:]), 0)
            # storing weights and biases
            layer_weights_1.append(WChild1)
            layer_weights_2.append(WChild2)
            layer_biases_1.append(bChild1)
            layer_biases_2.append(bChild2)
        return layer_weights_1, layer_weights_2, layer_biases_1, layer_biases_2

    def create_child(self, weights, biases):
        # randomly initiate child of correct size
        child = getSimpleNeuralNet(n_hidden=biases[0].shape[0])
        # copy over weights and biases
        for index, layer in enumerate(self.layer_names):
            with torch.no_grad():
                child[layer].weight.copy_(torch.tensor(weights[index], requires_grad=True))
                child[layer].bias.copy_(torch.tensor(biases[index], requires_grad=True))
        return child

    def matrix_example(self, W1, W2):
        # just an example to show how the recombination works
        cutoff = random.randint(1, min(W1.shape[0], W2.shape[0]))
        print(f'Cutoff is {cutoff}')
        Keep1 = W1[:cutoff, :cutoff]
        Keep2 = W2[:cutoff, :cutoff]
        SwapOutputs1 = W1[:cutoff, cutoff:]
        SwapOutputs2 = W2[:cutoff, cutoff:]
        SwapInputs1 = W1[cutoff:, :]
        SwapInputs2 = W2[cutoff:, :]
        WChild1 = np.concatenate((Keep1, SwapOutputs2), 1)
        WChild2 = np.concatenate((Keep2, SwapOutputs1), 1)
        WChild1 = np.concatenate((WChild1, SwapInputs2), 0)
        WChild2 = np.concatenate((WChild2, SwapInputs1), 0)
        print(WChild1)
        print(WChild2)

if __name__=='__main__':
    # Call this file as a script for testing
    net1 = getSimpleNeuralNet()
    net2 = getSimpleNeuralNet()

    recombiner = Recombiner()