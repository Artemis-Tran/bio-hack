# Computional Biologists' Society MiniHackathon 

## Description
Goal was to design a machine learning algorithm that would predict DNA sequence chromatin accessible sites from a background of random non-accessible sites in a cell type. Given three fasfa files, one containing 47,239 sequences that were accessible sites, one containing 478,499 sequences that were not accessible sites, and one containing 269,315 sequences that were randomly selected to be accessible or not accessible, our task was to output a file of 10,000 sequences that our algorithm classified as accessible from the third file.

To do so, we built a neural network containing linear layers and a ReLU layer to be trained on sequences from files 1 and 2, and classify sequences to be either accessible or not accessible from file 3. Finished with an honorable mention by acheiving a score above 5000.
