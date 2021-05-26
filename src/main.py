from ecomp import run_evolution
from configs import Configs
from datetime import datetime
import torch
from models import getSimpleNeuralNet, trainBaseNetwork, evaluateNetwork
from results import ResultsIO
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

    RETRAIN_BASE = False
    SAVE_BEST = True
    SAVE_RESULTS = True

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
        EPOCHS=1,
        LOSS_WEIGHTS=[0,2,1]
    )

    best_n, results = run_evolution(
        base_network,
        n_epochs=configs.EPOCHS,
        pop_size=configs.POP_SIZE,
        p_mut=configs.MUTATION_RATE,
        emb_layers=configs.EMB_LAYERS,
        recomb_layers=configs.RECOMBINATION_LAYERS,
        loss_weights=configs.LOSS_WEIGHTS,
        n_inputs=dim_num,
        n_outputs=n_classes,
        validation_loader=val_loader
    )

    save_format = "%Y-%m-%d--%H:%M:%S"
    save_name = datetime.now().strftime(save_format)

    evaluateNetwork(best_n, val_loader)
    if SAVE_BEST:
        console.log('Saving best network')
        torch.save(best_n.state_dict(), f'weights/best_net-{save_name}.pth')

    # Save results
    if SAVE_RESULTS:
        console.log('Saving results to disk')
        ResultsIO.save(
            path = 'results/',
            filename = save_name,
            configs = configs,
            results = results,
        )
    
    console.log('Done.')