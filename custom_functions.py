# Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder

import os
from typing import Dict, List, Tuple
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt

def set_seeds():
    """ Sets seeds for torch operations.
    
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

##########################################################################################################
##########################################################################################################

def open_random_image(path_list: List[Path],
                      seed: int = None) -> Image:
    """Opens random image from a path list with the meta data.
    
    Opens a random image from a path list and prints out
    the metadata given below:
        - mode
        - size
        - width
        - height
        - format
        - format_description

    Args:
        path_list(List[Path]): A list contains sequence of paths.
    """
    if seed:
        random.seed(seed)

    random_image_path = random.choice(seq = path_list)
    random_image_label = random_image_path.parent.stem
    image = Image.open(random_image_path)

    # Print the class label
    print(f"Class label : {random_image_label}")

    # Print the metadata
    print(f"Colour mode : {image.mode}")
    print(f"Size : {image.size}")
    print(f"With, Height : {image.width}, {image.height}")
    print(f"Image format : {image.format}")
    print(f"Format description : {image.format_description}")

    return image

##########################################################################################################
##########################################################################################################

def open_random_transformed_image(dataset: torchvision.datasets.folder.ImageFolder,
                                  transform: torchvision.transforms,
                                  seed: int = None):
    """Opens a random image and transforms it to see transformation effect.
    
    Args:
        dataset(torchvision.datasets.folder.ImageFolder): Dataset object in type of ImageFolder.
        transform(torchvision.transforms): Transforms object to be applied on random image.
    """
    # Set the seed value of seed is exists
    if seed:
        random.seed(seed)
    
    # Create class names
    class_names = dataset.classes

    # Create a random index
    random_idx = random.randint(a = 0,
                                b = len(dataset))
    
    # Take a random image and label
    random_image, random_label = dataset[random_idx][0], dataset[random_idx][1]

    # Transform image
    transformed_image = transform(random_image)

    # Show the transformed image
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.title(f"Class of the image : {class_names[random_label]}\nSize of the image {transformed_image.size()}")
    plt.axis(False)
    plt.show()

##########################################################################################################
##########################################################################################################

# Let's create a function to plot loss and accuracy curves.

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots loss and accuracy curves.

    Args:
        results(Dict[str, List[float]]): Results of the training in dictionary type.
    
    """

    # Get the loss values
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values
    train_acc = results['train_acc']
    test_acc = results['test_acc']

    # Extract the value of epochs
    epochs = range(len(train_loss))

    # Plot the loss curves
    plt.figure(figsize = (12, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label = "Train Loss")
    plt.plot(epochs, test_loss, label = "Test Loss")
    plt.xlabel("Epochs")
    plt.title("Loss Values")
    plt.legend()

    # Plot the accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label = "Train Accuracy")
    plt.plot(epochs, test_acc, label = "Test Accuracy")
    plt.xlabel("Epochs")
    plt.title("Accuracy Values")
    plt.legend();