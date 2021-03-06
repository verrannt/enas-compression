class Configs():

    def __init__(
        self,
        POP_SIZE,
        MUTATION_RATE,
        EMB_LAYERS,
        RECOMBINATION_LAYERS,
        EPOCHS,
        LOSS_WEIGHTS
    ):
        # Size of the population of compressed networks to optimize
        self.POP_SIZE = POP_SIZE
        # Chance of random mutation at each iteration of the algorithm
        self.MUTATION_RATE = MUTATION_RATE
        # Which layers to use when computing the similarity loss
        self.EMB_LAYERS = EMB_LAYERS
        # Which layers to use for the recombiner
        self.RECOMBINATION_LAYERS = RECOMBINATION_LAYERS
        # Number of epochs to run the optimization for
        self.EPOCHS = EPOCHS
        # Weights for the different loss functions, in the following order:
        # Accuracy, Embedding loss, Compression loss
        self.LOSS_WEIGHTS = LOSS_WEIGHTS

    def to_dict(self):
        return {
            'pop_size': self.POP_SIZE,
            'mutation_rate': self.MUTATION_RATE,
            'emb_layers': self.EMB_LAYERS,
            'recomb_layers': self.RECOMBINATION_LAYERS,
            'epochs': self.EPOCHS,
            'loss_weights': self.LOSS_WEIGHTS,
        }

    @staticmethod
    def from_dict(dict):
        return Configs(
            POP_SIZE = dict['pop_size'],
            MUTATION_RATE = dict['mutation_rate'],
            EMB_LAYERS = dict['emb_layers'],
            RECOMBINATION_LAYERS = dict['recomb_layers'],
            EPOCHS = dict['epochs'],
            LOSS_WEIGHTS = dict['loss_weights'],
        )

    def __repr__(self):
        return "Configs: \n"\
            f"    pop_size:       {self.POP_SIZE}\n"\
            f"    mutation_rate:  {self.MUTATION_RATE}\n"\
            f"    emb_layers:     {self.EMB_LAYERS}\n"\
            f"    recomb_layers:  {self.RECOMBINATION_LAYERS}\n"\
            f"    epochs:         {self.EPOCHS}\n"\
            f"    loss_weights:   {self.LOSS_WEIGHTS}\n"