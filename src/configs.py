class Configs():

    def __init__(
        self,
        POP_SIZE,
        MUTATION_RATE,
        EMB_LAYERS,
        RECOMBINATION_LAYERS,
        MAX_ITER,
        LOSS_WEIGHTS
    ):

        self.POP_SIZE = POP_SIZE
        self.MUTATION_RATE = MUTATION_RATE
        self.EMB_LAYERS = EMB_LAYERS
        self.RECOMBINATION_LAYERS = RECOMBINATION_LAYERS
        self.MAX_ITER = MAX_ITER
        # Weights for the different loss functions, in the following order:
        # Accuracy, Embedding loss, Compression loss
        self.LOSS_WEIGHTS = LOSS_WEIGHTS