from Bio import SeqIO

# record_dict =
# print(record_dict["accessible13"])  # use any record ID

#!/usr/bin/env python
import sys
import Bio
from Bio import SeqIO, SeqFeature
from Bio.SeqRecord import SeqRecord
import os

def cal_kmers(s, k):
    count_kmers = {}
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        if count_kmers.get(kmer) == None:
            count_kmers[kmer] = 1
        else:
            count_kmers[kmer] += 1
    return count_kmers

from Bio import SeqIO

kmers_list = []  # Setup an empty list
for record in SeqIO.parse("Files/accessible.fasta", "fasta"):
    kmers_list.append(cal_kmers(record.seq, 2))

print("Found %i entries" % len(kmers_list))
print(kmers_list[0])