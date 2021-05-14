import random
from warnings import filterwarnings
import numpy as np
import DistanceLoss from losses
import get_embeddings from embeddings

def initializer(base_model, pool_size):    
    init_pool = []
    #TODO: create pool of initial children using the SVM technique THIJME
    return init_pool

def mutator(pop, p_mut):
    new_pop = []
    for model in pop:
        if random.random() < p_mut:
            mutated_model = None #TODO: mutate the model ??
            model = mutated_model
        new_pop.append(model)        
    return new_pop()

def calc_fitnesses(base_networks, base_embeddings, pool, dist, data):
    fitnesses = []
    for model in pool:
        embedding_layers = ["1", "2"] #TODO: ????? PASCAL
        model_embeddings = get_embeddings(data, model, embedding_layers) #TODO: implement calculate_embedding PASCAL
        loss = dist(base_embeddings, model_embeddings) #TODO: fix loss function to list of embeddings PASCAL
        accuracy_measure = None #TODO: pass validation set, predict it and normalize it STIJN
        compression_measure = None #TODO: calculate % of size compared to parent network STIJN
        a, b, c = 1
        fitness = a * loss + b * accuracy_measure + c * compression_measure #TODO: calculate fitness ??
        fitnesses.append(fitness)
    pool_fitnesses_zipped = list(zip(pool, fitnesses))
    return pool_fitnesses_zipped

def crossover(parent1, parent2):
    child = None #TODO: implement cross-over THIJME / PASCAL
    return child

def selector_and_breeder(pop_fitnesses_zipped, mating_pool_size):
    pop, fitnesses = zip(*pop_fitnesses_zipped)
    mating_pool = np.random.choice(pop, mating_pool_size, fitnesses)
    np.random.shuffle(mating_pool)
    new_pop = []
    while len(mating_pool > 1):
        n1 = mating_pool.pop()
        n2 = mating_pool.pop()
        nc1, nc2 = crossover(n1, n2)
        new_pop.append(nc1)
        new_pop.append(nc2)

def main(base_network, max_iter, pop_size, p_mut, validation_dataset):
    dist = DistanceLoss()
    embedding_layers = ["1", "2"] #TODO: ????? PASCAL
    base_embeddings = get_embeddings(validation_dataset, base_network, embedding_layers)
    
    avg_fitnesses = []
    
    init_pop = initializer(base_network, pop_size)
    pop_fitnesses = calc_fitnesses(base_network, base_embeddings, init_pop, dist, validation_dataset)
    pop, fitnesses = zip(*pop_fitnesses)
    best_i = np.argmax(fitnesses)
    best_f = fitnesses[best_i]
    avg_fitnesses.append(np.average(fitnesses))
    best_n = pop[best_i]
    i = 0
    while i < max_iter: #TODO: specify more convergence criteria
        pop = selector_and_breeder(pop_fitnesses, pop_size)
        pop = mutator(pop, p_mut)
        pop_fitnesses = calc_fitnesses(base_network, base_embeddings, pop, dist, validation_dataset)
        pop, fitnesses = zip(*pop_fitnesses)
        best_i = np.argmax(fitnesses)
        best_f = fitnesses[best_i]
        avg_fitnesses.append(np.average(fitnesses))
        best_n = pop[best_i]
        i += 1
    return best_n, best_f, avg_fitnesses