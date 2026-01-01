import torch
from torch.utils.data import Dataset


class HousePriceDataset(Dataset):
    """
    Sets the required structure for the dataset used for training
    the neural network.
    Has 3 main methods:
     - init -> initialize the dataset
     - len  -> retrieve the length of the dataset
     - get  -> retrieve the dataset entry at any given index
    """

    def __init__(self, x, y) -> None:
        """
        Creates dataset using the given x and y values where
        x corresponds to the input given to nnet and y corresponds
        to the output.

        :param x: input data given to neural network
        :param y: expected output from the neural network
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        """
        returns the length of the dataset -> useful for training nnet
        (batching, num of data entries, etc.)

        :return: length of the dataset
        :rtype: int
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        gets the x and y values corresponsing to index

        :param idx: index of the data entry
        :return: x and y values at index idx of the dataset
        """
        x = self.x[idx]
        y = self.y[idx]
        return x, y

