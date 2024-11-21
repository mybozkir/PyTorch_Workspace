# Libraries

import torch
from torch import nn

from typing import Dict, Tuple, List
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

# Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"



def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               device: torch.device = device) -> Tuple[float, float]:
    """Training step for image classification.

    Args:
        model (nn.Module): Model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader object for training.
        optimizer (torch.optim.Optimizer): Optimizer object for training.
        loss_fn (nn.Module): Loss function.
        device (torch.device | optional): Device to be used during training.
    
    Returns:
        train_loss, train_acc(Tuple[float, float]): Average train loss and train accuracy values.
    """

    
    # Put model in train mode
    model.train()

    # Set the initial values of loss and accuracy
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Forward pass
        train_pred_logits = model(X)

        # Calculate loss
        loss = loss_fn(train_pred_logits, y)
        train_loss += loss.item()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy
        train_pred_labels = torch.argmax(torch.softmax(train_pred_logits, dim = 1), dim = 1)
        train_acc += (train_pred_labels == y).sum().item() / len(train_pred_logits)
    
    # Calculate the loss and acc per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Return the results
    return train_loss, train_acc

##########################################################################################################
##########################################################################################################

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device = device) -> Tuple[float, float]:
    """Test step for image classification.

    Args:
        model (nn.Module): Model to be tested.
        dataloader (torch.utils.data.DataLoader): DataLoader object for testing.
        loss_fn (nn.Module): Loss function.
        device (torch.device | optional): Device to be used during testing.
    
    Returns:
        test_loss, test_acc(Tuple[float, float]): Average test loss and test accuracy values.
    """
    
    # Put the model evaluation mode
    model.eval()

    # Set the initial values of test_loss and test_acc
    test_loss, test_acc = 0, 0

    # Loop through with inference mode
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim = 1), dim = 1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
    
    # Calculate the loss and acc per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # Return the results
    return test_loss, test_acc

##########################################################################################################
##########################################################################################################

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          device: torch.device = device) -> Dict[str, List[float]]:
    """Training and testing for image classification.

    Args:
        model (nn.Module): Model to be used for train and test.
        train_dataloader (torch.utils.data.DataLoader): DataLoader object for training.
        test_dataloader (torch.utils.data.DataLoader): DataLoader object for testing.
        optimizer (torch.optim.Optimizer): Optimizer object for training.
        loss_fn (nn.Module): Loss function.
        writer (torch.utils.tensorbard.writer.SummaryWriter): SummaryWriter object.
        device (torch.device | optional): Device to be used during training and testing.
        
    
    Returns:
        results(Dict[str, List[float]]): results dictionary contains loss and accuracy values for both training and testing.
    """
    
    results = {
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'test_loss': []
    }

    for epoch in tqdm(range(epochs)):
        
        # Evaluate train step
        train_loss, train_acc = train_step(model = model,
                                           dataloader = train_dataloader,
                                           optimizer = optimizer,
                                           loss_fn = loss_fn,
                                           device = device)
        # Evaluate test step
        test_loss, test_acc = test_step(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device)
        
        # Print the progress
        print(
            f"Epoch : {epoch + 1} | "
            f"Train Accuracy : {train_acc} | "
            f"Test Accuracy : {test_acc} | "
            f"Train Loss : {train_loss} | "
            f"Test Loss : {test_loss}"
        )

        # Update the results dictionary
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        # Set the summary writer
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag = "Accuracy",
                               tag_scalar_dict = {'train_acc' : train_acc,
                                                  'test_acc' : test_acc},
                                global_step = epoch)
            
            writer.add_scalars(main_tag = "Loss",
                               tag_scalar_dict = {'train_loss' : train_loss,
                                                  'test_loss' : test_loss},
                                global_step = epoch)
            writer.close()
        else:
            pass
        
    # Return the results dict
    return results