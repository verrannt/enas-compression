import torch
from torch import nn
from collections import OrderedDict

def getSimpleNeuralNet(dim_num=784, n_hidden=256, n_classes=10):
    """
    Return a simple PyTorch Sequential Neural Network.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("flatten", nn.Flatten()),
                ("layer_1_linear", nn.Linear(dim_num, n_hidden)),
                ("layer_1_act", nn.ReLU()),
                # ("layer_1_dropout", nn.Dropout(.1)),
                ("layer_2_linear", nn.Linear(n_hidden, n_hidden)),
                ("layer_2_act", nn.ReLU()),
                # ("layer_2_dropout", nn.Dropout(.1)),
                ("output", nn.Linear(n_hidden, n_classes)),
            ]
        )
    )

def trainBaseNetwork(network, train_loader, val_loader, n_epochs=4):

    print('Training base network')

    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    train_losses = []
    for t in range(n_epochs):
        print("Epoch {}/{}".format(t+1,n_epochs))
        for step, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            pred = network(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

    evaluateNetwork(network, val_loader)

    return network
        
def evaluateNetwork(network, val_loader):

    print('Evaluating best network')
    print(network)

    size = len(val_loader.dataset)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X,y in val_loader:
            # Compute prediction and loss
            pred = network(X)
            #test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    #test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")