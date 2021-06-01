import sys

from ecomp import run_evolution
from configs import Configs
from datetime import datetime
import torch
from models import getSimpleNeuralNet, trainBaseNetwork, evaluateNetwork
from results import ResultsIO
from data.fashion_mnist import FashionMNISTLoader
from rich.console import Console

console = Console()

def main(
    run_idx:int = 1, 
    exp_name:str = 'unnamed'
):
    train_loader, val_loader, dim_num = FashionMNISTLoader.get(
        train_size = 10000, 
        test_size = 10000
    )
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

    console.print(configs)

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
        train_loader=train_loader
    )

    save_format = "%Y-%m-%d_%H-%M-%S"
    save_name = datetime.now().strftime(save_format)
    save_name += f'_{exp_name}'
    save_name += f'_{run_idx}'

    print()
    console.log('Evaluating best network')
    best_test_acc = evaluateNetwork(best_n, val_loader)
    if SAVE_BEST:
        console.log('Saving best network')
        torch.save(best_n.state_dict(), f'weights/best-net_{save_name}.pth')

    # Save results
    if SAVE_RESULTS:
        console.log('Saving results to disk')
        ResultsIO.save(
            path = 'results/',
            filename = save_name,
            configs = configs,
            results = results,
            best_test_acc = best_test_acc
        )
    
    console.log('Done.')

if __name__=='__main__':
    try:
        n_runs = int(sys.argv[1])
        exp_name = sys.argv[2]
        console.log(f'Creating {sys.argv[1]} total runs.')
    except:
        raise ValueError(
            'Failed to read arguments. Please ensure that you called the '
            'script in the following manner: \n'
            '$ python src/main.py <n_runs> <exp_name>'
        )

    for run_idx in range(1,n_runs+1):
        console.print(f'\n[yellow]Run {run_idx} of {n_runs}')
        main(run_idx, exp_name)