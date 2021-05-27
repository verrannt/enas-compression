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
from results import ResultsManager
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

def calc_fitnesses(base_embeddings, pool, batch_data, batch_labels, base_size, emb_layers, loss_weights):

    fitnesses, accuracies, losses, comps = [], [], [], []

    # Initialize loss with base networks embeddings
    loss_fn = TSNELoss(base_embeddings[0])
    loss_fn_2 = TSNELoss(base_embeddings[1])

    for model in pool:
        model_embeddings = get_embeddings(batch_data, model, emb_layers)
        loss = loss_fn(model_embeddings[0]) + loss_fn_2(model_embeddings[1])
        accuracy = accuracy_measure(model, batch_data, batch_labels)
        compression = compression_measure(model, base_size)
        fitness = loss_weights[0] * accuracy \
                - loss_weights[1] * loss \
                + loss_weights[2] * (1-compression)
        fitnesses.append(fitness)
        accuracies.append(accuracy)
        losses.append(loss)
        comps.append(compression)

    loss_dict = {
        'fitnesses': fitnesses,
        'accuracies': accuracies,
        'losses': losses, 
        'comps': comps,

        'avg_fitness': np.average(fitnesses),
        'best_fitness': np.max(fitnesses),
        'worst_fitness': np.min(fitnesses),

        'avg_acc': np.average(accuracies),
        'best_acc': np.max(accuracies),
        'worst_acc': np.min(accuracies),

        'avg_loss': np.average(losses),
        'best_loss': np.min(losses),
        'worst_loss': np.max(losses),

        'avg_comp': np.average(comps),
        'best_comp': np.min(comps),
        'worst_comp': np.max(comps),
    }

    return loss_dict

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
    n_epochs, 
    pop_size, 
    p_mut, 
    emb_layers,
    recomb_layers,
    loss_weights,
    n_inputs,
    n_outputs,
    train_loader,
):

    console = Console()

    console.log('Initializing algorithm')
    recomb = Recombiner(recomb_layers, n_inputs, n_outputs)
    initialiser = Initialiser()
    mutator = NetworkMutation(recomb_layers, p_mut)

    console.log('Computing embeddings')
    # Get images from the dataset, ignore labels here
    batch_data, batch_labels = next(iter(train_loader))
    # Get embeddings of base network
    base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
    # Get total number of trainable parameters for base network
    base_size = count_parameters(base_network)

    results = ResultsManager()

    console.log('Initializing population')
    pop = initialise(initialiser, base_network, pop_size)

    # Calculate fitnesses for the first run. The `loss_dict` holds the results
    # of this single run and is overwritten each iteration.
    console.log('Computing initial population fitness')
    loss_dict = calc_fitnesses(
        base_embeddings, 
        pop, 
        batch_data,
        batch_labels, 
        base_size, 
        emb_layers,
        loss_weights,
    )

    # Track results
    results.add(loss_dict)

    console.log('Starting optimization')
    for epoch in range(n_epochs):
        i = 0
        pbar = Progbar(len(train_loader))
        console.print(f'\n[blue]Epoch {epoch+1}/{n_epochs}')
        for batch_data, batch_labels in train_loader:

            # Get embeddings of base network
            base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
            
            # Create new population
            pop = selector_and_breeder(pop, loss_dict['fitnesses'], pop_size, recomb)
            pop = mutator(pop)

            # Calculate fitnesses for this iteration. The `loss_dict` holds the
            # results of this iteration only and is overwritten each iteration.
            loss_dict = calc_fitnesses(
                base_embeddings, 
                pop, 
                batch_data,
                batch_labels, 
                base_size, 
                emb_layers,
                loss_weights,
            )

            # Get best individual
            best_i = np.argmax(loss_dict['fitnesses'])
            best_n = pop[best_i]
            
            # Track results        
            results.add(loss_dict)

            i += 1
            pbar.update(i, [
                ('Fitness', loss_dict['avg_fitness']), 
                #('Best', loss_dict['best_fitness']), 
                #('Worst', loss_dict['worst_fitness']), 
                ('Acc', loss_dict['avg_acc']), 
                #('Best Acc', loss_dict['best_acc']), 
                #('Worst Acc', loss_dict['worst_acc']),
                ('Loss', loss_dict['avg_loss']), 
                #('Best Loss', loss_dict['best_loss']), 
                #('Worst Loss', loss_dict['worst_loss']),
                ('Comp', loss_dict['avg_comp']), 
                #('Best Comp', loss_dict['best_comp']), 
                #('Worst Comp', loss_dict['worst_comp']),
            ])
        del pbar

    #Compute final fitnesses
    console.print('Computing final fitnesses')
    final_fitnesses = []

    i = 0
    pbar = Progbar(len(train_loader))
    for batch_data, batch_labels in train_loader:
        # Get embeddings of base network
        base_embeddings = get_embeddings(batch_data, base_network, emb_layers)
        loss_dict = calc_fitnesses(
            base_embeddings,
            pop,
            batch_data,
            batch_labels,
            base_size,
            emb_layers,
            loss_weights,
        )
        final_fitnesses.append(loss_dict['fitnesses'])
        i += 1
        pbar.update(i)

    del pbar
    avg_losses = np.average(final_fitnesses, axis=0)

    # get best individual
    best_i = np.argmax(avg_losses)
    best_n = pop[best_i]

    print()
    return best_n, results