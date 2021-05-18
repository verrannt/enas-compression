import random
import numpy as np
import torch
from models import getSimpleNeuralNet

class NetworkMutation():

    def __init__(self, layer_names, probability):
        self.layer_names = layer_names
        self.probability = probability

    def _mutate_model(self, model):
        """
        Mutate a given PyTorch model by adding or subtracting 10% from weights.
        """
        model_dict = dict(model.named_children())
        for layer in self.layer_names:
            layer_weights = model_dict[layer].weight.detach().numpy()
            layer_shape = layer_weights.shape
            noise_range = np.random.uniform(0.9, 1.1, layer_shape)
            with torch.no_grad():
                model_dict[layer].weight.copy_(torch.tensor(
                    layer_weights * noise_range, requires_grad=True))
        return model

    def __call__(self, population):
        """
        Mutate models in a population with certain probability
        """
        new_pop = []
        for model in population:
            if random.random() < self.probability:
                model = self._mutate_model(model)
            new_pop.append(model)
        return new_pop

if __name__=='__main__':
    # Test this module
    pop = [getSimpleNeuralNet(),getSimpleNeuralNet(),getSimpleNeuralNet()]
    mutator = NetworkMutation(
        layer_names=["layer_1_linear", "layer_2_linear", "output"],
        probability=0.3
    )
    model = mutator(pop)