import json
from configs import Configs

import numpy as np

class Metric:
    """
    Keeps track of metrics with internal lists representing
    averages, best, and worst values across time.
    """

    def __init__(self, name):
        self.name = name
        self.averages = []
        self.best = []
        self.worst = []

    def from_dict(self, dict):
        """
        Set internal values from provided dict. Requires dict to have
        'averages', 'best', and 'worst' keys, otherwise will throw
        KeyError.
        """
        self.averages = dict['averages']
        self.best = dict['best']
        self.worst = dict['worst']

        return self

    def to_dict(self):
        """
        Return dictionary from internal values, with 'averages', 'best',
        and 'worst' keys and corresponding list value pairs.
        """
        return {
            'averages': self.averages,
            'best': self.best,
            'worst': self.worst,
        }

class ResultsManager:

    def __init__(self):
        self.fitness = Metric('Fitness')
        self.accuracy = Metric('Accuracy')
        self.loss = Metric('Loss')
        self.compression = Metric('Compression')

    def add(self, loss_dict):
        # Averages
        self.fitness.averages.append(loss_dict['avg_fitness'])
        self.accuracy.averages.append(loss_dict['avg_acc'])
        self.loss.averages.append(loss_dict['avg_loss'])
        self.compression.averages.append(loss_dict['avg_comp'])
        # Best
        self.fitness.best.append(loss_dict['best_fitness'])
        self.accuracy.best.append(loss_dict['best_acc'])
        self.loss.best.append(loss_dict['best_loss'])
        self.compression.best.append(loss_dict['best_comp'])
        # Worst
        self.fitness.worst.append(loss_dict['worst_fitness'])
        self.accuracy.worst.append(loss_dict['worst_acc'])
        self.loss.worst.append(loss_dict['worst_loss'])
        self.compression.worst.append(loss_dict['worst_comp'])

    def to_dict(self):
        return {
            'fitness': self.fitness.to_dict(),
            'accuracy': self.accuracy.to_dict(),
            'loss': self.loss.to_dict(),
            'compression': self.compression.to_dict()
        }

    @staticmethod
    def from_dict(dict):
        # Init empty results
        results = ResultsManager()
        # Overwrite metric values
        results.fitness = Metric('Fitness').from_dict(
            dict['fitness'])
        results.accuracy = Metric('Accuracy').from_dict(
            dict['accuracy'])
        results.loss = Metric('Loss').from_dict(
            dict['loss'])
        results.compression = Metric('Compression').from_dict(
            dict['compression'])

        return results

    def compute_metrics_string(self, list):
        return f"{len(list)} values, min: {np.min(list):.3f}, max: {np.max(list):.3f}, mean: {np.mean(list):2.3f}"

    def __str__(self):
        return "Results:\n"\
        f"    Fitness:\n"\
        f"        Averages: {self.compute_metrics_string(self.fitness.averages)}\n"\
        f"        Best:     {self.compute_metrics_string(self.fitness.best)}\n"\
        f"        Worst:    {self.compute_metrics_string(self.fitness.worst)}\n"\
        f"    Accuracy:\n"\
        f"        Averages: {self.compute_metrics_string(self.accuracy.averages)}\n"\
        f"        Best:     {self.compute_metrics_string(self.accuracy.best)}\n"\
        f"        Worst:    {self.compute_metrics_string(self.accuracy.worst)}\n"\
        f"    Loss:\n"\
        f"        Averages: {self.compute_metrics_string(self.loss.averages)}\n"\
        f"        Best:     {self.compute_metrics_string(self.loss.best)}\n"\
        f"        Worst:    {self.compute_metrics_string(self.loss.worst)}\n"\
        f"    Compression:\n"\
        f"        Averages: {self.compute_metrics_string(self.compression.averages)}\n"\
        f"        Best:     {self.compute_metrics_string(self.compression.best)}\n"\
        f"        Worst:    {self.compute_metrics_string(self.compression.worst)}\n"\

class ResultsIO:

    @staticmethod
    def save(
        path, 
        filename, 
        configs, 
        results, 
        best_test_acc
    ):
        save_dict = {
            'name': filename,
            'configs': configs.to_dict(),
            'results': results.to_dict(),
            'best_test_acc': best_test_acc
        }
        with open(path+filename+'.json', 'w') as f:
            json.dump(save_dict, f)

    @staticmethod
    def load(path, filename):
        with open(path+filename+'.json', 'r') as f:
            unparsed_dict = json.load(f)

        configs = Configs.from_dict(
            unparsed_dict['configs']
        )
        results = ResultsManager.from_dict(
            unparsed_dict['results']
        )

        best_test_acc = unparsed_dict['best_test_acc']

        return configs, results, best_test_acc