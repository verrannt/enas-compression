import torch_pruning as tp
import copy

class Initialiser():
    """
    Enables initialisation of compressed networks
    """

    def __init__(self):
        pass

    def __call__(self, network, compression_rate):
        # TODO: add retraining if we want to
        return self.compress(network, compression_rate)

    def compress(self, network, compression_rate):
        network_copy = copy.deepcopy(network)

        # pruning according to L1 Norm
        strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()

        # Build dependency graph
        DG = tp.DependencyGraph()
        # example_inputs should be of correct input size
        DG.build_dependency(network_copy, example_inputs=torch.randn(1, 784))

        # get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
        pruning_plan = DG.get_pruning_plan(network_copy[1], tp.prune_linear,
                                           idxs=strategy(network_copy[1].weight, amount=compression_rate))
        pruning_plan_2 = DG.get_pruning_plan(network_copy[3], tp.prune_linear,
                                             idxs=strategy(network_copy[3].weight, amount=compression_rate))

        # execute this plan (prune the model)
        pruning_plan.exec()
        pruning_plan_2.exec()

        return network_copy

