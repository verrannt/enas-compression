import random
from warnings import filterwarnings
import numpy as np
from losses import DistanceLoss
from mutation import NetworkMutation
from recombiner import Recombiner
from initialiser import Initialiser
from embeddings import get_embeddings
from measures import accuracy_measure, compression_measure

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def initialise(
    initialiser, 
    base_model, 
    pool_size, 
    compression_min=0.1, 
    compression_max = 0.8
):
    init_pool = []
    for compression_rate in np.arange(
                                compression_min, 
                                compression_max, 
                                ((compression_max-compression_min)/pool_size)
                            ):
        init_pool.append(initialiser(base_model, compression_rate))
    return init_pool

def calc_fitnesses(base_embeddings, pool, loss_fn, dataset, base_size, emb_layers):
    batch_data, batch_labels = dataset[0], dataset[1]
    fitnesses = []
    for model in pool:
        model_embeddings = get_embeddings(batch_data, model, emb_layers)
        loss = loss_fn(model_embeddings[0])
        accuracy = accuracy_measure(model, batch_data, batch_labels)
        compression = compression_measure(model, base_size)
        a, b, c = 1, 1, 1
        fitness = a * accuracy + b * (1-loss) + c * compression #TODO: calculate fitness ??
        fitnesses.append(fitness)
    #pool_fitnesses_zipped = list(zip(pool, fitnesses))
    return fitnesses

def selector_and_breeder(pop, fitnesses, mating_pool_size, recombiner):
    #pop, fitnesses = zip(*pop_fitnesses_zipped)
    mating_idx = np.random.choice(
        range(len(pop)), mating_pool_size, p=softmax(fitnesses), replace=True)
    np.random.shuffle(mating_idx)
    mating_pool = [pop[i] for i in mating_idx]
    new_pop = []
    while len(mating_pool) > 1:
        n1 = mating_pool.pop()
        n2 = mating_pool.pop()
        # recombiner does the crossover
        nc1, nc2 = recombiner(n1, n2)
        #TODO: Do we need to select before adding to new pop?
        new_pop.append(nc1)
        new_pop.append(nc2)
    return new_pop

def run_evolution(
    base_network, 
    max_iter, 
    pop_size, 
    p_mut, 
    emb_layers,
    recomb_layers,
    n_inputs,
    n_outputs,
    validation_dataset,
):

    dist = DistanceLoss()
    recomb = Recombiner(recomb_layers, n_inputs, n_outputs)
    initialiser = Initialiser()
    mutator = NetworkMutation(recomb_layers, p_mut)

    # Get images from the dataset, ignore labels here
    batch_data, _ = validation_dataset
    # Get embeddings of base network
    base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
    # Get total number of trainable parameters for base network
    base_size = sum(p.numel() for p in base_network.parameters() if p.requires_grad)

    # Initialize loss with base networks embeddings
    # TODO right now only takes one layer
    emb_loss = TSNELoss(base_embeddings[0])

    avg_fitnesses = []
    pop = initialise(initialiser, base_network, pop_size)
    fitnesses = calc_fitnesses(
        base_embeddings, 
        pop, 
        emb_loss, 
        validation_dataset, 
        base_size, 
        emb_layers
    )
    #pop, fitnesses = zip(*pop_fitnesses)
    best_i = np.argmax(fitnesses)
    best_f = fitnesses[best_i]
    avg_fitnesses.append(np.average(fitnesses))
    best_n = pop[best_i]
    i = 0
    while i < max_iter: #TODO: specify more convergence criteria
        print(f"Iteration {i+1}/{max_iter}\r", end='')
        pop = selector_and_breeder(pop, fitnesses, pop_size, recomb)
        pop = mutator(pop)
        fitnesses = calc_fitnesses(
            base_embeddings, 
            pop, 
            emb_loss, 
            validation_dataset, 
            base_size, 
            emb_layers
        )
        best_i = np.argmax(fitnesses)
        best_f = fitnesses[best_i]
        avg_fitnesses.append(np.average(fitnesses))
        best_n = pop[best_i]
        i += 1
    return best_n, best_f, avg_fitnesses