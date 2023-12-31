import sys
import Bio
from Bio import SeqIO, SeqFeature
import os
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt

# defined macros
K_VALUE = 2
ACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("accessible.fasta")
NOTACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("notaccessible.fasta")
TEST_FASTA_FILE_PATH = os.path.abspath("test.fasta")

### PRE-PROCESSING DATA

def cal_kmers(s, k):
    count_kmers = {}
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        if count_kmers.get(kmer) == None:
            count_kmers[kmer] = 1
        else:
            count_kmers[kmer] += 1
    return count_kmers

# Setup an empty list
kmers_list = []  
for record in SeqIO.parse(ACCESSIBLE_FASTA_FILE_PATH, "fasta"):
    kmers_list.append(cal_kmers(record.seq, K_VALUE))

MAX_LENGTH = 999999
for record in SeqIO.parse(NOTACCESSIBLE_FASTA_FILE_PATH, "fasta"):
    kmers_list.append(cal_kmers(record.seq, K_VALUE))
    if len(kmers_list) > MAX_LENGTH:
        break

test_kmers_list = []
test_sequence_ids = []
for record in SeqIO.parse(TEST_FASTA_FILE_PATH, "fasta"):
    test_kmers_list.append(cal_kmers(record.seq, K_VALUE))
    test_sequence_ids.append(record.id)

matrix = pd.DataFrame(kmers_list)
matrix.fillna(0, inplace=True)

test_matrix = pd.DataFrame(test_kmers_list)
test_matrix.fillna(0, inplace=True)

accessible_labels = [0] * (47239)
non_accessible_labels = [1] * (len(kmers_list) - 47239)
labels = accessible_labels + non_accessible_labels

X_train, X_test, y_train, y_test = train_test_split(matrix, labels, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

print("Found %i entries" % len(kmers_list))
print(matrix)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix.png')

test_predictions = clf.predict(test_matrix)

output_file = "top_accessible_sites.txt"
with open(output_file, 'w') as file:
    count = 0
    for sequence_id, prediction in zip(test_sequence_ids, test_predictions):
        if prediction == 0:  # predicted as accessible
            file.write(f"{sequence_id}\n")
            count += 1
            if count >= 10000:
                break
