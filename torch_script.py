import sys
import Bio
from Bio import SeqIO, SeqFeature
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUE = 2
MAX_LENGTH = 299999
ACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("accessible.fasta")
NOTACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("notaccessible.fasta")
TEST_FASTA_FILE_PATH = os.path.abspath("test.fasta")

# Class Model
class sequence_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=16, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=128)
        self.layer_3 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    

### PRE-PROCESSING DATA

# function to count and calculate kmers for a given sequence
def cal_kmers(s, k):
    count_kmers = {}
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        if count_kmers.get(kmer) == None:
            count_kmers[kmer] = 1
        else:
            count_kmers[kmer] += 1
    return count_kmers

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

### TRAINING MODEL
def prepare_data():
    print("Preparing Data...")
    #filling list with kmer calculations for each sequence
    kmers_list = []  
    for record in SeqIO.parse(ACCESSIBLE_FASTA_FILE_PATH, "fasta"):
        kmers_list.append(cal_kmers(record.seq, K_VALUE))
    accessible_labels = [0] * len(kmers_list)

    
    for record in SeqIO.parse(NOTACCESSIBLE_FASTA_FILE_PATH, "fasta"):
        kmers_list.append(cal_kmers(record.seq, K_VALUE))
        if len(kmers_list) > MAX_LENGTH:
            break
    non_accessible_labels = [1] *(len(kmers_list) - len(accessible_labels))

    all_keys = sorted(set().union(*[d.keys() for d in kmers_list]))
    ordered_kmers_list = [{key: d.get(key, 0) for key in all_keys} for d in kmers_list]
    y = np.array(accessible_labels + non_accessible_labels)
    kmers_list_numeric = [list(count_kmers.values()) for count_kmers in ordered_kmers_list]

    X = np.array(kmers_list_numeric)
    matrix = pd.DataFrame(X)
    print(matrix)
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    print(X)
    print(y[:10])
    print(X.shape)
    print(y.shape)
    
    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size=0.2, random_state=42)
    
    # Model to train
    model_0 = sequence_model().to(DEVICE)
    

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

    # logits -> prediction probablities -> prediction lablels
    # y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(DEVICE))))

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Number of iterations 
    epochs = 1000

    # Putting data to right device
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE) 
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE) 

    for epoch in range(epochs):
        ### Training
        model_0.train()

        # Forward Pass
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate Loss
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true = y_train, y_pred = y_pred)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backwards pass
        loss.backward()

        # Step
        optimizer.step()

        model_0.eval()
        with torch.inference_mode():
            # Forward Pass
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            # Calculate Loss
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    return model_0

def predict_test(model):
    test_kmers_list = []
    test_sequence_ids = []
    for record in SeqIO.parse(TEST_FASTA_FILE_PATH, "fasta"):
        test_kmers_list.append(cal_kmers(record.seq, K_VALUE))
        test_sequence_ids.append(record.id)
    all_keys = sorted(set().union(*[d.keys() for d in test_kmers_list]))
    ordered_kmers_list = [{key: d.get(key, 0) for key in all_keys} for d in test_kmers_list]
    kmers_list_numeric = [list(count_kmers.values()) for count_kmers in ordered_kmers_list]

    X = np.array(kmers_list_numeric)
    X = torch.from_numpy(X).type(torch.float).to(DEVICE)
    model.eval()
    with torch.inference_mode():
        predictions = torch.round(torch.sigmoid(model(X)))
    print(len(predictions))
    predictions = torch.Tensor.cpu(predictions)
    predictions = torch.Tensor.numpy(predictions)

    output_file = "predictions.csv"
    with open(output_file, 'w') as file:
        count = 0
        for sequence_id, prediction in zip(test_sequence_ids, predictions):
            if prediction[0] == 0:  # predicted as accessible
                file.write(f"{sequence_id}\n")
                count += 1
                if count >= 10000:
                    break
    return predictions

    # Evaluation metric
    

if __name__ == "__main__":
    model = prepare_data()
    print(predict_test(model))





