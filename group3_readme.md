# Transformer Based Fully Connected Neural Network for Chemical Property Prediction
This repository contains a PyTorch implementation of a fully connected neural network (FCNN) combined with a Transformer encoder for predicting chemical properties based on molecular fingerprints.
## Requirements
* Python 3.6+
* PyTorch 1.8+
* RDKit
* Pandas
* Numpy
## Installation


To install the required dependencies, you can use the following command:

```bash
pip install torch pandas numpy rdkit-pypi
```
## Dataset
The model expects a CSV file with at least two columns: one for the SMILES representation of the molecules and another for the property to be predicted. The CSV file should be formatted as follows:

| SMILES  | docking score |
|---------|---------------|
| c1cccc  | 1.23          |
| c3saac  | 1.10          |

## Usage
1. Prepare your dataset in the format mentioned above and save it as test.csv.
2. Run the main script:
```python [fcnn_transformer_train_test.py](fcnn_transformer_train_test.py)```
The script will train the model on the provided dataset and print the training and testing losses. The trained model will be saved as fully_connected_model.pth.
## Model
The model consists of a fully connected neural network (FCNN) followed by a Transformer encoder. The FCNN processes the input molecular fingerprints, and the Transformer encoder captures the sequential relationships between the patches of the fingerprint.
## Customization
- You can customize the model by adjusting the following parameters:

  - input_dim: The dimensionality of the input fingerprint (default: 1024).
  - hidden_dim: The dimensionality of the hidden layer in the Transformer encoder (default: 512).
  - num_classes: The number of classes or the output dimension (default: 1 for regression tasks).
  - batch_size: The batchsize of training and testing (default: 16).

