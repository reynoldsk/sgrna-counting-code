#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:46:17 2021
edited: 04/30/21
edited: 08/27/21 Ryan Otto
edited: 02/26/24 Phil Brown
edited: 07/23/24 Phil Brown
edited: 08/02/25 Phil Brown
edited: 08/13/15 Phil Brown

@author: PhilBrown


This script reads a single file name from the command line and counts
each of the sgRNA pairs and their barcodes from the fastq files.

This is for paired end sequencing of the plasmids, where one read contains
the first sgRNA in the plasmid and the second contains a second sgRNA.

Illumina quality score cheatsheet: https://help.basespace.illumina.com/files-used-by-basespace/quality-scores
"""

# import useful modules/packages
# import glob
import regex as re
import numpy as np
import pandas as pd
import sys
from Bio import SeqIO
from itertools import islice
from Bio.Seq import Seq
import logging
import datetime
import gzip
from contextlib import contextmanager
from pathlib import Path


def get_logger(filename: str | None = None) -> logging.Logger:
    """
    Initializes a logger for logging information about the counting.
    Defaults to stdout stream, otherwise uses the provided filename to log to a file.

    Parameters
    -------

    filename : str | None
        The name of the file to log to. If None, logs to stdout.

    Returns
    -------

    logging.Logger
        A logger object for logging information.
    """
    
    if filename is not None:
        open(filename, "a").close()  # Clear the log file if it already exists
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
        )
    logger = logging.getLogger(__name__)
    return logger


def build_guide_dict(
    fasta_file: str,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """
    From the fasta file, take the sequences and their id to create a dictionary of sgRNA id to
    homology sequence and homology sequence to sgRNA id, as well as a list of the sgRNA ids.

    Parameters
    -----------
    fasta_file : str
        The path to the fasta file containing the sgRNA sequences and their ids.

    Returns
    --------
    tuple[dict[str, str], dict[str, str], list[str]]
        A tuple containing:
        - A dictionary mapping sgRNA ids to their corresponding homology sequences (sgRNA2seq).
        - A dictionary mapping homology sequences to their corresponding sgRNA ids (seq2sgRNA).
        - A list of sgRNA ids (guides_list).
    """

    sgRNA2seq: dict[str, str] = {}
    seq2sgRNA: dict[str, str] = {}

    # find homology + reverse complement of homology from PW_Folate_corrected.fa
    for record in SeqIO.parse(fasta_file, "fasta"):
        homologous_seq = str(record.seq[35:55].reverse_complement())
        seq2sgRNA[homologous_seq] = str(record.id)
        sgRNA2seq[str(record.id)] = homologous_seq
    guides_list = list(sgRNA2seq.keys())
    return sgRNA2seq, seq2sgRNA, guides_list


def build_quality_check_dict(position_file: str) -> dict[str, list[str]]:
    """
    Build a dictionary of crucial nucleotide positions for quality checking in filtering step.
    The nucleotides at these positions are used to differentiate similar homology regions from each other.
    If the quality score of these positions are below the threshold, we cannot be confident in the identity of the sgRNA.

    Parameters
    ----------
    position_file : str
        The path to the file containing the crucial nucleotide positions for quality checking.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping sgRNA ids to a list of crucial nucleotide positions for q-score checking checking.
    """

    nt_sgRNA: dict[str, list[str]] = {}
    # read files with positions to check in q30 filter step
    with open(position_file) as file:
        line = file.readline()
        while line:
            line = line.strip("\n")
            line_split = line.split(":")
            nt_sgRNA[line_split[0]] = []
            number_nt = line_split[1].split(",")
            nt_sgRNA[line_split[0]].append(number_nt[0])
            nt_sgRNA[line_split[0]].append(number_nt[1])
            line = file.readline()
    return nt_sgRNA


def qscore_filter(seq_qscores: str, qthreshold: int = 30) -> bool:
    """
    Check if the quality scores given are above the threshold.

    Parameters
    ----------
    seq_qscores : str
        A string of quality scores for the crucial nucleotide positions for a given sgRNA.
    qthreshold : int, optional
        The quality score threshold for filtering reads. Default is 30,
        which means only reads with quality scores above 30 will pass the filter.

    Returns
    -------
    bool
        True if all quality scores are above the threshold, False otherwise.
    """
    high_quality = True
    for base in seq_qscores:
        ## q scores are encoded as ASCII + 33
        if (ord(base) - 33) <= qthreshold:
            high_quality = False
            break
    return high_quality


def seq_find(
    seq: str, qscore: str, left: str, right: str, target_len: int, hamming_dist: int
) -> tuple[str, str]:
    """
    Find the target sequence and corresponding quality scores bounded by the left and right sequences,
    allowing for a certain hamming distance (number of mismatches).

    Parameters
    ------------
    seq: string
        The sequence to search within.
    qscore: string
        The quality scores of each nucleotide corresponding to the sequence.
    left: string
        The left boundary sequence to search for.
    right: string
        The right boundary sequence to search for.
    target_len: int
        The expected length of the target sequence between the left and right boundaries.
    hamming_dist: int
        The maximum number of mismatches allowed in the left and right boundaries (the target sequence can be anything).

    Returns
    -----------
    tuple[str, str]
        A tuple containing:
        - The target sequence found between the left and right boundaries, or "Not found" if no unique match is found.
        - The corresponding quality scores for the target sequence, or "Not found" if no unique match is found.
    """

    make_string = (
        "(" + left + "." * target_len + right + ")" + "{s<" + str(hamming_dist) + "}"
    )
    matches = re.finditer(make_string, str(seq))
    for i, match in enumerate(matches):
        ## if more than one match, we return "Not found"; cannot be sure which one to use
        if i > 0:
            target_seq = "Not found"
            target_qscore = "Not found"
            return target_seq, target_qscore
        found_seq = seq[match.start() : match.end()]
        found_qscore = qscore[match.start() : match.end()]
    try:
        target_seq = found_seq[len(left) : -len(right)]
        target_qscore = found_qscore[len(left) : -len(right)]
    except (IndexError, UnboundLocalError):
        target_seq = "Not found"
        target_qscore = "Not found"
    return target_seq, target_qscore


def check_qscore(
    nt_sgRNA: dict[str, list[str]],
    sgRNA1_name: str,
    qscore_sgRNA1: str,
    sgRNA2_name: str,
    qscore_sgRNA2: str,
    qthreshold: int = 30,
) -> tuple[bool, bool]:
    """
    Check the whether the qscores of a read are above the threshold.

    Parameters
    ----------
    nt_sgRNA: dict[str, list[str]]
        A dictionary mapping sgRNA homology regions to a
        list of crucial nucleotide positions for q-score checking.
    sgRNA1_name: str
        The homology region of the first sgRNA to check.
    qscore_sgRNA1: str
        The quality scores for the first sgRNA region.
    sgRNA2_name: str
        The homology region of the second sgRNA to check.
    qscore_sgRNA2: str
        The quality scores for the second sgRNA region.
    qthreshold: int, optional
        The quality score threshold for filtering reads.
        Default is 30, which means only reads with quality
        scores above 30 will pass the filter.

    Returns
    -------
    tuple[bool, bool]
        A tuple containing:
        - A boolean indicating whether the quality scores for the first sgRNA are above the threshold.
        - A boolean indicating whether the quality scores for the second sgRNA are above the threshold.
    """

    if sgRNA1_name in nt_sgRNA.keys():
        if sgRNA1_name in nt_sgRNA.keys():
            for nt1 in nt_sgRNA[sgRNA1_name]:
                a = qscore_filter(qscore_sgRNA1[int(nt1) - 1], qthreshold)
                if not a:
                    break
            for nt2 in nt_sgRNA[sgRNA2_name]:
                b = qscore_filter(qscore_sgRNA2[int(nt2) - 1], qthreshold)
                if not b:
                    break
    else:
        a = False
        b = False
    return a, b


def get_files(file_name: str) -> tuple[str, str]:
    """
    Get the file names for the forward and backward reads based on the provided file name.

    Parameters
    ----------
    file_name : str
        The base file name to construct the read file names from.

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - The file name for the forward read (read_1_file).
        - The file name for the backward read (read_2_file).
    """
    read_1_file = f"{file_name}_R1_001.fastq"
    read_2_file = f"{file_name}_R2_001.fastq"

    if not Path(read_1_file).is_file():
        read_1_file += ".gz"
    if not Path(read_2_file).is_file():
        read_2_file += ".gz"

    return read_1_file, read_2_file


@contextmanager
def open_file(file_name: str):
    """
    Custom context manager for opening a file, 
    handling both regular and gzipped files.

    Parameters
    ----------
    file_name : str
        The name of the file to open.
    
    Yields
    -------
    f: file object
        The opened file object, which can be used within the context.
    """
    if file_name.endswith(".gz"):
        f = gzip.open(file_name, "rt")  # Open gzipped file in text mode
        try:
            yield f
        finally:
            f.close()
    else:
        f = open(file_name, "r")  # Open regular file
        try:
            yield f
        finally:
            f.close()


def build_sgRNA_lists(
    logger: logging.Logger,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], int]:
    # function used for identifying the sgRNAs present in reads from HiSeq
    """
    Build lists containing:
    - sgRNA ids for each forward and backward read of the plasmids
    - BC sequences, and Q scores for forward and backward read of the plasmids

    Requires the file_name from the command line.

    Parameters:
    ----------
    logger: logging.Logger
        A logger object for logging information about the counting process.

    Returns:
    -------
    tuple[list[str], list[str], list[str], list[str], list[str], list[str], int]
        A tuple containing:
        - A list of sgRNA ids for the forward reads (sgRNA1_list).
        - A list of quality scores for the forward reads (qscore_sgRNA1_list).
        - A list of sgRNA ids for the backward reads (sgRNA2_list).
        - A list of quality scores for the backward reads (qscore_sgRNA2_list).
        - A list of BC sequences found in each read (BC_list).
        - A list of quality scores corresponding to the BC sequences (qscore_BC_list).
        - The total number of reads processed (read_total).

    Each list is ordered such that the sgRNA ids, BC sequences,
    and quality scores at the same index correspond to the same read.
    """

    read_total = 0

    read_file = file_name
    logger.info(f"Processing file: {read_file}")

    read_1_file, read_2_file = get_files(file_name)

    logger.info(read_file)

    sgRNA1_list: list[str] = []
    qscore_sgRNA1_list: list[str] = []
    sgRNA2_list: list[str] = []
    qscore_sgRNA2_list: list[str] = []
    BC_list: list[str] = []
    qscore_BC_list: list[str] = []

    # one of these is the forward read and one is the backward read of an overlapping region
    with open_file(read_1_file) as fwdFile:
        while True:
            # skipping forward 4 lines because of fastq formatting
            next_n_lines = list(islice(fwdFile, 4))
            if not next_n_lines:
                break
            sequence = next_n_lines[1]
            q_score = next_n_lines[3]
            read_total += 1

            sgRNA2_seq, qscore_sgRNA2 = seq_find(
                sequence, q_score, "CTAGCTCTAAAAC", "A", 20, 3
            )
            BC_seq, qscore_BC = seq_find(
                sequence, q_score, "GTACAGCGAGGCAAC", "ACGGATCCCCAC", 6, 3
            )
            sgRNA2_list.append(sgRNA2_seq)
            qscore_sgRNA2_list.append(qscore_sgRNA2)
            BC_list.append(BC_seq)
            qscore_BC_list.append(qscore_BC)

    # sgRNA2 is the bwd read I think
    with open_file(read_2_file) as bwdFile:
        while True:
            next_n_lines = list(islice(bwdFile, 4))
            if not next_n_lines:
                break
            sequence = Seq(next_n_lines[1]).reverse_complement()
            q_score = next_n_lines[3][::-1]
            read_total += 1
            sgRNA1_seq, qscore_sgRNA1 = seq_find(
                sequence, q_score, "CTAGCTCTAAAAC", "A", 20, 3
            )
            sgRNA1_list.append(sgRNA1_seq)
            qscore_sgRNA1_list.append(qscore_sgRNA1)

    return (
        sgRNA1_list,
        qscore_sgRNA1_list,
        sgRNA2_list,
        qscore_sgRNA2_list,
        BC_list,
        qscore_BC_list,
        read_total,
    )


def get_overall_counts(
    sgRNA1_list: list[str],
    sgRNA2_list: list[str],
    BC_list: list[str],
    logger: logging.Logger,
) -> tuple[dict[str, int], int]:
    """
    Get overall counts of reads with sgRNA1, sgRNA2, and BC found for the file.
    This is just for logging purposes.

    Parameters:
    ----------
    sgRNA1_list: list[str]
        A list of sgRNA ids for the forward reads (sgRNA1).
    sgRNA2_list: list[str]
        A list of sgRNA ids for the backward reads (sgRNA2).
    BC_list: list[str]
        A list of barcode (BC) sequences found in each read.
    logger: logging.Logger
        A logger object for logging information about the counting process.

    Returns:
    -------
    tuple[dict[str, int], int]
        A tuple containing the reads dictionary with counts of
        sgRNA1, sgRNA2, BC, total reads, and successful reads
        and the total number of reads with all sequences found.
    """
    all_seqs_total = 0
    reads_dict = {}

    sgRNA1_sum = sum([x != "Not found" for x in sgRNA1_list])
    sgRNA2_sum = sum([x != "Not found" for x in sgRNA2_list])
    BC_sum = sum([x != "Not found" for x in BC_list])
    total = len(sgRNA1_list)

    logger.info(f"sgRNA1 found: {sgRNA1_sum}")
    logger.info(f"sgRNA2 found: {sgRNA2_sum}")
    logger.info(f"BC found: {BC_sum}")
    logger.info(f"Total reads: {total}")

    both = [
        len(x) == 20 and len(y) == 20 and len(z) == 6
        for x, y, z in zip(sgRNA1_list, sgRNA2_list, BC_list)
    ]
    reads_dict["sgRNA1"] = sgRNA1_sum
    reads_dict["sgRNA2"] = sgRNA2_sum
    reads_dict["BC"] = BC_sum
    reads_dict["total"] = total
    reads_dict["success"] = sum(both)

    Success_print = "Success: " + str(sum(both))
    logger.info(Success_print)

    all_seqs_total += sum(both)

    return reads_dict, all_seqs_total


def build_sgRNA_count_dicts(
    guides_list: list[str],
    sgRNA1_list: list[str],
    qscore_sgRNA1_list: list[str],
    sgRNA2_list: list[str],
    qscore_sgRNA2_list: list[str],
    counted_BC_list: list[str],
    nt_sgRNA: dict[str, list[str]],
    seq2sgRNA: dict[str, str],
    qthreshold: int,
    logger: logging.Logger,
    BC_list: list[str],
) -> None:
    """
    Build counts matrices for each barcode. Runs in place on
    a pairwise_lib dictionary, which is used for saving unfiltered counts.
    Also logs the counts of reads that pass the q-score filtering step.

    Parameters:
    ----------
    guides_list: list[str]
        A list of sgRNA ids.
    sgRNA1_list: list[str]
        A list of sgRNA ids for the forward reads.
    qscore_sgRNA1_list: list[str]
        A list of quality scores for the forward reads.
    sgRNA2_list: list[str]
        A list of sgRNA ids for the backward reads.
    qscore_sgRNA2_list: list[str]
        A list of quality scores for the backward reads.
    counted_BC_list: list[str]
        A list of barcode (BC) sequences found in each read.
    nt_sgRNA: dict[str, list[str]]
        A dictionary mapping sgRNA homology regions to a
        list of crucial nucleotide positions for q-score
        filtering.
    seq2sgRNA: dict[str, str]
        A dictionary mapping homology sequences to their
        corresponding sgRNA ids.
    qthreshold: int
        The quality score threshold for filtering reads.
    logger: logging.Logger
        A logger object for logging information about the
        counting process.
    BC_list: list[str]
        A list of barcode (BC) sequences to look for int he reads.
    """
    ab = 0
    onlya = 0
    onlyb = 0
    none = 0
    failed = 0
    ## seeds the pairwise_lib dictionaries
    for BC in BC_list + ["Other"]:
        pairwise_lib[BC] = pd.DataFrame(
            np.zeros([len(guides_list), len(guides_list)]),
            np.array(guides_list),
            np.array(guides_list),
        )

    ## populates the pairwise_lib dictionaries
    for i, guide1 in enumerate(sgRNA1_list):
        guide2 = sgRNA2_list[i]
        BC = counted_BC_list[i]
        if len(guide1) == 20 and len(guide2) == 20:
            if BC in BC_list:
                try:
                    guide1_id = seq2sgRNA[guide1]
                    guide2_id = seq2sgRNA[guide2]

                    a, b = check_qscore(
                        nt_sgRNA,
                        guide1_id,
                        qscore_sgRNA1_list[i],
                        guide2_id,
                        qscore_sgRNA2_list[i],
                        qthreshold=qthreshold,
                    )
                    if a and b:
                        pairwise_lib[BC].at[guide1_id, guide2_id] += 1
                        ab += 1
                    elif a and not b:
                        onlya += 1
                        continue
                    elif b and not a:
                        onlyb += 1
                        continue
                    else:
                        none += 1
                        continue
                except (KeyError, IndexError, UnboundLocalError):
                    failed += 1
            else:
                try:
                    guide1_id = seq2sgRNA[guide1]
                    guide2_id = seq2sgRNA[guide2]

                    pairwise_lib["Other"].at[guide1_id, guide2_id] += 1

                except (KeyError, IndexError, UnboundLocalError):
                    failed += 1
    sum_low = onlya + onlyb + none
    sum_all = sum_low + ab

    logger.info("qscore > " + str(qthreshold) + " for both sgRNAs: " + str(ab))
    logger.info(
        "qscore < " + str(qthreshold) + " for sgRNA in position 1: " + str(onlya)
    )
    logger.info(
        "qscore < " + str(qthreshold) + " for sgRNA in position 2: " + str(onlyb)
    )
    logger.info("qscore < " + str(qthreshold) + " for both sgRNAs: " + str(none))
    logger.info("sum < " + str(qthreshold) + ": " + str(sum_low))
    logger.info("all successful reads: " + str(sum_all))
    logger.info("total reads: " + str(sum_all + failed))


def reindex_pairwise_dict(guides_list: list[str]) -> None:
    """
    Reindex the pairwise dictionary in place
    by the given guide list and transposes the data vectors.

    Parameters:
    ----------
    guides_list: list[str]
        A list of guide sequences to use for reindexing the pairwise dictionary.

    """

    for BC in pairwise_lib.keys():
        guides_list.sort()
        pairwise_lib[BC] = pairwise_lib[BC][guides_list].T
        pairwise_lib[BC] = pairwise_lib[BC][guides_list].T


def save_unfiltered_counts(filepath: str = "BarcodedCounts/unfiltered/") -> None:
    """
    Saves the a matrix of counts for each barcode as a csv file in the filepath directory.

    Parameters:
    ----------
    filepath: str
        The directory path where the count matrices will be saved. Default is "BarcodedCounts/unfiltered/".
    """

    date = datetime.date.today()
    TP = file_name.split("/")[1]  ## file name
    for BC in pairwise_lib.keys():
        pd.DataFrame(pairwise_lib[BC]).to_csv(
            filepath + str(date) + "_" + TP + "_" + BC + "_counts.csv"
        )


if __name__ == "__main__":
    # logger = get_logger("counting_code_gz.log")

    qthreshold = 30

    # output dictionary
    pairwise_lib = {}
    BC_list = ["TGAAAG", "CATGAT", "CCATGC", "TCATAC", "TAGACT", "ACTAGG"]

    # read through a single file name
    file_name = sys.argv[1]
    logger = get_logger(f"counting_code_{file_name.split('/')[-1]}.log")

    logger.info(f"Currently processing: {file_name}")

    sgRNA2seq, seq2sgRNA, guides_list = build_guide_dict("PW_Folate_corrected.fa")
    nt_sgRNA = build_quality_check_dict("nt_sgRNA_FolateSinglesCorrected.txt")
    sgRNA1, qscore_sgRNA1, sgRNA2, qscore_sgRNA2, BC, qscore_BC, read_total = (
        build_sgRNA_lists(logger)
    )
    reads_dict, all_seqs_total = get_overall_counts(sgRNA1, sgRNA2, BC, logger)

    ## alters pairwise_lib in place, which is used for saving unfiltered counts
    build_sgRNA_count_dicts(
        guides_list,
        sgRNA1,
        qscore_sgRNA1,
        sgRNA2,
        qscore_sgRNA2,
        BC,
        nt_sgRNA,
        seq2sgRNA,
        qthreshold,
        logger,
        BC_list,
    )
    reindex_pairwise_dict(guides_list)
    save_unfiltered_counts()
