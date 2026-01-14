import torch
from torch import nn
from torch import device
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def model_training_epoch(model : nn.Module,
                         training_data_loader : DataLoader,
                         loss_function : nn.Module,
                         optimizer : Optimizer,
                         device : device):
    """
    One epoch of model training (One complete forward pass and back propagation step for all
    inputs and outputs in dataset)

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
    model.train()
    total_epoch_loss = 0.0

    for input_batch, output_batch in training_data_loader:
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        current_batch_loss = 0.0

        optimizer.zero_grad() #reset the calculated gradient to 0
        predictions = model(input_batch) #calculate the predicted output for current input batch
        loss = loss_function(predictions, output_batch) #calculate the difference between expected and predicted outputs
        loss.backward() #perform gradient
        optimizer.step()

        total_epoch_loss += abs(loss.item())

    average_epoch_loss = total_epoch_loss/len(training_data_loader)
    return average_epoch_loss

