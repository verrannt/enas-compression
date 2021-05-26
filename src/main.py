from ecomp import run_evolution
from configs import Configs
from datetime import datetime
import torch
from models import getSimpleNeuralNet, trainBaseNetwork, evaluateNetwork
from data.fashion_mnist import FashionMNISTLoader
from rich.console import Console

console = Console()

if __name__=='__main__':

    train_loader, val_loader, dim_num = FashionMNISTLoader.get()
    n_classes = 10

    base_network = getSimpleNeuralNet(
        dim_num=dim_num,
        n_hidden=256,
        n_classes=n_classes,
    )

    RETRAIN_BASE = True

    if RETRAIN_BASE:
        # Train base network
        console.log('Training base network')
        base_network = trainBaseNetwork(base_network, train_loader, val_loader)
        torch.save(base_network.state_dict(), 'weights/base_weights.pth')
    else:
        # Load weights from storage
        console.log('Loading base network from weights file')
        base_network.load_state_dict(torch.load('weights/base_weights.pth'))

    configs = Configs(
        POP_SIZE=100,
        MUTATION_RATE=0.01,
        EMB_LAYERS=["layer_1_act", "layer_2_act"],
        RECOMBINATION_LAYERS=["layer_1_linear", "layer_2_linear", "output"],
        MAX_ITER=100,
    )

    best_n, best_f, avg_fitnesses = run_evolution(
        base_network,
        max_iter=configs.MAX_ITER,
        pop_size=configs.POP_SIZE,
        p_mut=configs.MUTATION_RATE,
        emb_layers=configs.EMB_LAYERS,
        recomb_layers=configs.RECOMBINATION_LAYERS,
        n_inputs=dim_num,
        n_outputs=n_classes,
        validation_dataset=next(iter(val_loader))
    )

    evaluateNetwork(best_n, val_loader)
    torch.save(best_n.state_dict(), f'weights/best_net-{datetime.now()}.pth')