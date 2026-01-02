import torch
from torch import nn
from torch import device
from torch.utils.data import DataLoader

def model_validation_epoch(model : nn.Module,
                           validation_data_loader : DataLoader,
                           loss_function : nn.Module,
                           device : device):
    """
    Docstring for model_training_epoch

    :param model: The neural network to be trained
    :type model: nn.Module
    :param training_data_loader: The data loader module which loads the data in batches when required
    :type training_data_loader: DataLoader
    :param loss_function: Loss function for calculating the loss by comparing prediction and expected output
                          loss.backward() performs backpropagation
    :type loss_function: nn.Module (parent base class for loss functions)
    :param optimizer: To perform corrections according to calculated loss to get better outputs
                      Adjusts weights and biases during training to minimize the loss function
    :type optimizer: Optimizer
    :param device: The device on which the model training takes place
                   (cuda cores in nvidia gpu (if available) OR cpu)
    :type device: device
    """
    model.to(device=device)
    model.eval()
    total_epoch_loss = 0.0

    with torch.no_grad():
        for idx, (input_batch, output_batch) in enumerate(validation_data_loader):
            current_batch_loss = 0.0

            predictions = model(input_batch)
            loss = loss_function(predictions, output_batch)

            current_batch_loss += abs(loss.item())
            average_batch_loss = current_batch_loss/len(input_batch)
            #can print avg batch loss if needed
            total_epoch_loss += average_batch_loss

    average_epoch_loss = total_epoch_loss/len(validation_data_loader)
    return average_epoch_loss
