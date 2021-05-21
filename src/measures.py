import torch 

def test_loop(data, labels, model):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0
    predictions = []
    with torch.no_grad():
        # TODO Feed whole batch instead of iterating through it
        # TODO May need to put on device?
        for X, y in zip(data,labels):
            # Compute prediction and loss
            pred = model(X)
            predictions.append(pred)
            correct += (pred.argmax() == y.argmax()).type(torch.float).sum().item()

    correct /= len(data)
    return correct

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compression_measure(child_network, parent_size):
    child_size = count_parameters(child_network)
    return child_size / parent_size

def accuracy_measure(child_network, batch_data, batch_labels):
    return test_loop(batch_data, batch_labels, child_network)