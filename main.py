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
import joblib

# defined macros
K_VALUE = 3
ACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("accessible.fasta")
NOTACCESSIBLE_FASTA_FILE_PATH = os.path.abspath("notaccessible.fasta")
TEST_FASTA_FILE_PATH = os.path.abspath("test.fasta")

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

### TRAINING MODEL
def train_model():
    print("Training model...")
    #filling list with kmer calculations for each sequence
    kmers_list = []  
    for record in SeqIO.parse(ACCESSIBLE_FASTA_FILE_PATH, "fasta"):
        kmers_list.append(cal_kmers(record.seq, K_VALUE))
    MAX_LENGTH = 999999
    for record in SeqIO.parse(NOTACCESSIBLE_FASTA_FILE_PATH, "fasta"):
        kmers_list.append(cal_kmers(record.seq, K_VALUE))
        if len(kmers_list) > MAX_LENGTH:
            break

    # Creating 2d matrix with kmer counts
    print("Creating 2d matrix with kmer counts...")
    matrix = pd.DataFrame(kmers_list)
    matrix.fillna(0, inplace=True)

    # Creating approriate labels, 0 for accessible and 1 for not accessible
    print("Creating approriate labels, 0 for accessible and 1 for not accessible...")
    accessible_labels = [0] * (47239)
    non_accessible_labels = [1] * (len(kmers_list) - 47239)
    labels = accessible_labels + non_accessible_labels

    # Splitting the training data to training data and testing data to test accuracy
    print("Splitting the training data to training data and testing data to test accuracy...")
    X_train, X_test, y_train, y_test = train_test_split(matrix, labels, test_size=0.2)

    # Initializing and fitting the random forest to the training data
    print("Initializing and fitting the random forest to the training data...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)


    # Number of trees in random forest
    n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


    print("Fitting data...")
    clf.fit(X_train, y_train)

    # Making predictions on the testing set
    print("Making predictions on the testing set...")
    y_pred = clf.predict(X_test)

    # Making confusion matrix png
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix.png')

    # save the model to a file
    model_filename = "random_forest_model.joblib"
    joblib.dump(clf, model_filename)
    print(f"Trained model saved to {model_filename}")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return clf

### PREDICTING RESULTS
def predict_test(clf):

    #filling list with kmer calculations for each sequence and creating corresponding list of ids
    test_kmers_list = []
    test_sequence_ids = []
    for record in SeqIO.parse(TEST_FASTA_FILE_PATH, "fasta"):
        test_kmers_list.append(cal_kmers(record.seq, K_VALUE))
        test_sequence_ids.append(record.id)

    # Create 2D matrix 
    test_matrix = pd.DataFrame(test_kmers_list)
    test_matrix.fillna(0, inplace=True)

    # make predictions based on training
    test_predictions = clf.predict(test_matrix)

    # making file with our predicted accessible sites
    output_file = "predictions.csv"
    with open(output_file, 'w') as file:
        count = 0
        for sequence_id, prediction in zip(test_sequence_ids, test_predictions):
            if prediction == 0:  # predicted as accessible
                file.write(f"{sequence_id}\n")
                count += 1
                if count >= 10000:
                    break

if __name__ == "__main__":
    clf = train_model()
    predict_test(clf)