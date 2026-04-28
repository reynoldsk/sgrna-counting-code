from itertools import islice
from pathlib import Path

"""
Script to get all file-headers up to read number 
and write them to a text file. This is used for 
parallelization of the counting code and the format
of the list is compatible with the code.

Usage 
------

Path should be a relative path to the directory 
containing the fastq.gz files.

Fasta_list should be the name of the text file to which
the file headers will be written. Each line of the text 
file will be a file header, which can be used as an input
to the counting code.
"""

if __name__ == "__main__":

    path = "crispri-data"
    fasta_list = "fasta_list.txt"

    files = []
    for file in Path(path).rglob("*.fastq.gz"):
        files.append(str(file))
    
    file_names = []
    for file in files:
        file_names.append(file.split("_R")[0])

    file_names = set(file_names)
    with open(fasta_list, "w") as f:
        for file in file_names:
            f.write(file + "\n")