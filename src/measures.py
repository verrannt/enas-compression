from torch import nn
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def test_loop(data, labels, model):
    correct = 0
    predictions = []
    with torch.no_grad():
        for index, datapoint in enumerate(data):
            # Compute prediction and loss
            X = torch.tensor(datapoint).type(torch.FloatTensor).to(device) #input
            y = torch.tensor(labels[index, :]).type(torch.FloatTensor).to(device) #labels
            pred = model(X)
            predictions.append(pred)
            correct += (pred.argmax() == y.argmax()).type(torch.float).sum().item()

    correct /= len(data)
    return 100*correct

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compression_measure(child_network, parent_size):
    child_size = count_parameters(child_network)
    return child_size / parent_size

def accuracy_measure(child_network, batch_data, batch_labels):
    return test_loop(batch_data, batch_labels, child_network)