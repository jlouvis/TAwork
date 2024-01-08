import os
import pytest
from torchvision.datasets import MNIST
# from your_module import data_utils
from tests import _PATH_DATA
import torchvision.transforms as transforms
from torchvision import datasets

import torch

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

def test_data():
    # Assuming data_utils provides a function to load the dataset
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

    # Check the number of samples in the dataset
    expected_train_samples = 60000  # Update based on your requirement
    expected_test_samples = 5000    # Update based on your requirement
    assert len(trainset) == expected_train_samples
    # assert len(dataset) == expected_test_samples

    # Check the shape of each datapoint
    for data, label in trainset:
        assert data.shape == torch.Size([1, 28, 28]) or data.shape == torch.Size([784])
        assert label >= 0 and label <= 9  # Assuming MNIST labels are in the range [0, 9]

    # for data, label in dataset:
    #     assert data.shape == torch.Size([1, 28, 28]) or data.shape == torch.Size([784])
    #     assert label >= 0 and label <= 9

if __name__ == "__main__":
    pytest.main()
