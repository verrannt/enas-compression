from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

class FashionMNISTLoader():

    def get():
        """
        Get dataloaders for the FashionMNIST train and test set from the 
        PyTorch repositories, and return them together with the dimension 
        required for a neural network's input layer.
        """

        training_data = datasets.FashionMNIST(
            root="../data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="../data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        train_dataloader = DataLoader(
            training_data, 
            batch_size=64, 
            shuffle=True
        )
        test_dataloader = DataLoader(
            test_data, 
            batch_size=64, 
            shuffle=True
        )

        dim_num = training_data[0][0].size()[1] * training_data[0][0].size()[2]
        
        return train_dataloader, test_dataloader, dim_num