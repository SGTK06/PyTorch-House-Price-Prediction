import torch
from torch import nn
from torch import device
from torch.utils.data import DataLoader

def model_validation_epoch(model : nn.Module,
                           validation_data_loader : DataLoader,
                           loss_function : nn.Module,
                           device : device):
    """
    Docstring for model_validation_epoch

    :param model: The neural network to be trained
    :type model: nn.Module
    :param training_data_loader: The data loader module which loads the data in batches when required
    :type training_data_loader: DataLoader
    :param loss_function: Loss function for calculating the loss by comparing prediction and expected output
                          loss.backward() performs backpropagation
    :type loss_function: nn.Module (parent base class for loss functions)
    :param device: The device on which the model training takes place
                   (cuda cores in nvidia gpu (if available) OR cpu)
    :type device: device
    """
    model.to(device=device)
    model.eval()
    total_epoch_loss = 0.0

    with torch.no_grad():
        for input_batch, output_batch in validation_data_loader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            predictions = model(input_batch)
            loss = loss_function(predictions, output_batch)

            total_epoch_loss += loss.item()

    average_epoch_loss = total_epoch_loss/len(validation_data_loader)
    return average_epoch_loss
