#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import csv


# In[ ]:


# Model Preparation adn Data Loading
class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe.iloc[:, 1]  # The first column is the label
        self.inputs = dataframe.iloc[:, 2].values  # The second column is the input

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
            if input_tensor.shape[0] != 1024:
                # If data dimension is not 1024, return a default 1024-dimensional zero vector
                input_tensor = torch.zeros(1024, dtype=torch.float32)
            return input_tensor, label
        except Exception as e:
            # If an exception occurs during data transformation, return default value or error label
            print(f"Error at index {idx}: {e}")  # Print error message
            default_label = torch.tensor(0, dtype=torch.float32)  # Default label value
            default_input = torch.zeros(10, dtype=torch.float32)  # Assume input length is a zero vector of 10
            return default_input, default_label
class Fcnn_Transformer(nn.Module): # Transformer block
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=1, patch_size=8):
        super(Fcnn_Transformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) * (32 // patch_size)
        self.embedding_dim = input_dim // self.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.patch_embedding = nn.Linear(self.embedding_dim, hidden_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.encoder_block = nn.TransformerEncoderLayer(hidden_dim, nhead=1)
        self.fc_0 = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Insert one layer of the FullyConnectedNN here

    def forward(self, x):
        x = self.fc_0(x) # Pass through the fully connected layer
        batch_size = x.size(0)
        x = x.view(batch_size, 64 // self.patch_size, 64 // self.patch_size, self.num_patches) # torch.Size([16, 8, 8, 16])
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)  # B, len, patch_size, patch_size
        # print(x.shape)
        x = x.flatten(2)  # B,  len, patch_size*patch_size
        # print(x.shape)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # B, 1, hidden_dim  Classify token
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # B, num_patches + 1, hidden_dim
        x += self.positional_embedding

        x = self.encoder_block(x)
        cls_output = x[:, 0]  # Extracting the output of the cls token
        output = self.fc(cls_output)
        return output.squeeze(1)


# In[ ]:


### *** Clean the data and extract the columns we need *** ###
import csv
def extract_and_save_columns(input_file_path, output_file_directory, column_names_to_extract):
    # Build output file path
    output_file_path = f"{output_file_directory}/extracted_columns.csv"

    with open(input_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Ensure specified column names exist
        missing_columns = [col for col in column_names_to_extract if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Columns not found in CSV: {missing_columns}")

        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=column_names_to_extract)
            writer.writeheader()
            for row in reader:
                #  Write only the columns we are interested in
                writer.writerow({col: row[col] for col in column_names_to_extract})

    return output_file_path

# Specify the path to the original CSV file
input_csv_file_path = '/Users/Downloads/4511 Proj/SP_7JXQ_A_no-H2O_1cons_M1-div_arm_hb_16rota_new-smiles_dedup_full_columns.csv' 
# Specify the directory where the new CSV file will be saved (saved an extracted_columns.csv file for future use)
output_directory = '/Users/Downloads/4511 Proj'

# Column names to extract
columns_to_extract = ['SMILES', 'docking score']

# Execute function and create a new CSV file
try:
    extract_columns_csv_path = extract_and_save_columns(input_csv_file_path, output_directory, columns_to_extract)
    print(f"New CSV with selected columns created at: {extract_columns_csv_path}")
except ValueError as e:
    print(e)


# In[ ]:


### *** Process data into fingerprint and load data into the model *** ###
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    # Read data and process into fingerprint
    df1 = pd.read_csv(extract_columns_csv_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    df = df1
    # Remove rows where the 'SMILES' column contains NaN
    df_clean = df.dropna(subset=['SMILES'])
    # Fingerprint
    mol_list = []
    for smiles in df_clean['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_list.append(mol)

    fingerprint_arrays = []
    for mol in mol_list:
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp_arr = np.zeros((1024,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            fingerprint_arrays.append(fp_arr)

    # Ensure df_clean and fingerprint_arrays are of the same length
    assert len(df_clean) == len(fingerprint_arrays)
    # Convert the list of arrays into a format suitable for a DataFrame column
    df_clean['fingerprint_arrays'] = fingerprint_arrays
    
    # Load data
    dataset = MyDataset(df_clean)
    # Calculate the number of data points in the training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Set parameters for DataLoader
    batch_size = 4096
    batch_size_test = 128
    shuffle = True
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Create DataLoader objects for the training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    # Set input/output dimensions and hidden layer dimensions
    input_dim = 1024
    hidden_dim = 512
    output_dim = 1
    num_epochs = 100
    #  Instantiate the model
    model = Fcnn_Transformer(input_dim, hidden_dim, output_dim).to(device)
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print the model structure
    print(model)

    # Train
    # for epoch in range(num_epochs):
    #     i = 0
    #     for inputs, labels in train_loader:
    #         # Forward propagation
    #         output = model(inputs.to(device))
    #         # Calculate loss
    #         loss = criterion(output.view(-1, 1).to(device), labels.view(-1, 1).to(device))
    #         # Back propagation
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         i+=1
    #         if i % 1000==0:
    #             print(i,': ',loss)
    #     if (epoch + 1) % 1 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    #
    # print('Training finished.')
    #
    # # Save Model
    # torch.save(model.state_dict(), 'fully_connected_model.pth')

    # Test
    model.eval()  # Set model to evaluation mode
    model.load_state_dict(torch.load('fully_connected_model.pth'))
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            test_loss += criterion(torch.squeeze(outputs).to(device), labels.view(-1, 1).to(device)).item()

    test_loss /= len(test_loader)*batch_size_test
    print(f'Test Loss: {test_loss:.4f}')

    print('Training and testing finished.')

