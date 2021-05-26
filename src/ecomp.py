import random
from warnings import filterwarnings
import numpy as np
from losses import DistanceLoss, TSNELoss
from mutation import NetworkMutation
from recombiner import Recombiner
from initialiser import Initialiser
from embeddings import get_embeddings
from measures import accuracy_measure, compression_measure, count_parameters
from utils import timer
from tensorflow.keras.utils import Progbar
from rich.console import Console

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

def calc_fitnesses(base_embeddings, pool, batch_data, batch_labels, base_size, emb_layers):

    fitnesses, accuracies, losses, comps = [], [], [], []

    # Initialize loss with base networks embeddings
    # TODO right now only takes one layer
    loss_fn = TSNELoss(base_embeddings[0])
    loss_fn_2 = TSNELoss(base_embeddings[1])

    for model in pool:
        model_embeddings = get_embeddings(batch_data, model, emb_layers)
        loss = loss_fn(model_embeddings[0]) + loss_fn_2(model_embeddings[1])
        accuracy = accuracy_measure(model, batch_data, batch_labels)
        compression = compression_measure(model, base_size)
        a, b, c = 0, 2, 1
        fitness = a * accuracy - b * loss + c * (1-compression) #TODO: calculate fitness ??
        fitnesses.append(fitness)
        accuracies.append(accuracy)
        losses.append(loss)
        comps.append(compression)
    #print(np.unique(comps, return_counts=True))
    #pool_fitnesses_zipped = list(zip(pool, fitnesses))
    #avg_acc = np.average(accuracies)
    #best_acc = accuracies[np.argmax(accuracies)]
    #worst_acc = accuracies[np.argmin(accuracies)]
    #avg_loss = np.average(losses)
    #best_loss = losses[np.argmin(losses)]
    #worst_loss = losses[np.argmax(losses)]
    #avg_comp = np.average(comp)
    #best_comp  = comps[np.argmin(comps)]
    #worst_comp = comps[np.argmax(comps)]

    loss_dict = {
        'avg_acc': np.average(accuracies),
        'best_acc': accuracies[np.argmax(accuracies)],
        'worst_acc': accuracies[np.argmin(accuracies)],
        'avg_loss': np.average(losses),
        'best_loss': losses[np.argmin(losses)],
        'worst_loss': losses[np.argmax(losses)],
        'avg_comp': np.average(comps),
        'best_comp': comps[np.argmin(comps)],
        'worst_comp': comps[np.argmax(comps)],
    }

    return fitnesses, accuracies, losses, comps, loss_dict, #avg_acc, best_acc, worst_acc, avg_loss, best_loss, worst_loss, avg_comp, best_comp, worst_comp

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

@timer
def run_evolution(
    base_network, 
    max_iter, 
    pop_size, 
    p_mut, 
    emb_layers,
    recomb_layers,
    n_inputs,
    n_outputs,
    validation_loader,
):

    console = Console()

    console.log('Initializing algorithm')
    recomb = Recombiner(recomb_layers, n_inputs, n_outputs)
    initialiser = Initialiser()
    mutator = NetworkMutation(recomb_layers, p_mut)

    console.log('Computing embeddings')
    # Get images from the dataset, ignore labels here
    batch_data, batch_labels = next(iter(validation_loader))
    # Get embeddings of base network
    base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
    # Get total number of trainable parameters for base network
    base_size = count_parameters(base_network)

    avg_fitnesses = []
    console.log('Initializing population')
    pop = initialise(initialiser, base_network, pop_size)
    console.log('Computing initial population fitness')
    fitnesses, accuracies, losses, comps, loss_dict = calc_fitnesses(
        base_embeddings, 
        pop, 
        batch_data,
        batch_labels, 
        base_size, 
        emb_layers
    )
    #pop, fitnesses = zip(*pop_fitnesses)
    best_i = np.argmax(fitnesses)
    best_f = fitnesses[best_i]
    avg_fitnesses.append(np.average(fitnesses))
    best_n = pop[best_i]

    console.log('Starting optimization')
    n_epochs = 1
    for epoch in range(n_epochs):
        i = 0
        pbar = Progbar(max_iter)
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        for batch_data, batch_labels in validation_loader:
        #while i < max_iter: #TODO: specify more convergence criteria
            # Get images from the dataset
            #batch_data, batch_labels = next(iter(validation_loader))
            # Get embeddings of base network
            base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
            #print(f"Iteration {i+1}/{max_iter}\r", end='')
            pop = selector_and_breeder(pop, fitnesses, pop_size, recomb)
            pop = mutator(pop)

            fitnesses, accuracies, losses, comps, loss_dict = calc_fitnesses(
                base_embeddings, 
                pop, 
                batch_data,
                batch_labels, 
                base_size, 
                emb_layers
            )

            best_i = np.argmax(fitnesses)
            best_f = fitnesses[best_i]
            
            best_acc_i = np.argmax(accuracies)

            worst_i = np.argmin(fitnesses)
            worst_f = fitnesses[worst_i]
            
            avg_fitness = np.average(fitnesses)
            avg_fitnesses.append(avg_fitness)
            
            best_n = pop[best_acc_i]

            i += 1
            pbar.update(i, [
                ('Avg', avg_fitness), 
                ('Best', best_f), 
                ('Worst', worst_f), 
                #('Avg Acc', loss_dict['avg_acc']), 
                #('Best Acc', loss_dict['best_acc']), 
                #('Worst Acc', loss_dict['worst_acc']),
                ('Avg Loss', loss_dict['avg_loss']), 
                ('Best Loss', loss_dict['best_loss']), 
                ('Worst Loss', loss_dict['worst_loss']),
                ('Avg Comp', loss_dict['avg_comp']), 
                ('Best Comp', loss_dict['best_comp']), 
                ('Worst Comp', loss_dict['worst_comp']),
            ])
        del pbar
    return best_n, best_f, avg_fitnesses