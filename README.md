# ai-hw-summer-2026-nn
# MNIST Recognition by Neural Networks

## Problem

This project builds small neural networks for handwritten digit recognition using the MNIST dataset.

The goal is to classify each image into one of 10 digit classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.

## Dataset

The dataset is loaded using torchvision.datasets.MNIST.

The model is trained on the training set by using train=True.

The model is tested on the test set by using train=False.

The test set is not used for training.

## Models

This project includes two models:

1. Shallow Multi-Layer Perceptron (MLP)
2. Convolutional Neural Network (CNN)

## Results

| Model | Test Accuracy |
|---|---:|
| MLP | 97.24% |
| CNN | 98.84% |

## Output Files

- mnist_assignment.py: Python code for loading data, training models, and testing models
- mnist_nn_assignment.ipynb: Google Colab notebook
- summary.txt: Testing results
- sample_predictions.png: Sample prediction examples
- confusion_matrix.png: Confusion matrix from the CNN model

## How to Run

Install requirements:

pip install torch torchvision matplotlib

Run the code:

python mnist_assignment.py

## Notes

This project follows the assignment requirement:

- Train on the training part
- Test on the test part
- Upload both code and testing results to GitHub
