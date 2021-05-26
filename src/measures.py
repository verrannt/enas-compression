import torch 

def test_loop(data, labels, model):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        # TODO May need to put on device?
        predictions = model(data)
        correct = (predictions.argmax(1) == labels).type(torch.float).sum().item()
    correct /= len(data)
    return correct

def count_parameters(model):
    # NOTE right now we only take the number of neurons in one hidden layer.
    # This works because both hidden layers have the same number of params.
    # If in the future we train models with different layer sizes this has
    # to be changed.
    return model[1].weight.shape[0]

def compression_measure(child_network, parent_size):
    child_size = count_parameters(child_network)
    return child_size / parent_size

def accuracy_measure(child_network, batch_data, batch_labels):
    return test_loop(batch_data, batch_labels, child_network)