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


def get_logger(filename: str | None = None) -> logging.Logger:
    if filename is not None:
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
) -> tuple[dict[str, Seq], dict[Seq, str], list[str]]:
    """
    From the fasta file, take the sequences and their id to create a dictionary of of sgRNAs
    """

    sgRNA2seq = {}
    seq2sgRNA = {}

    # find homology + reverse complement of homology from PW_Folate_corrected.fa
    for record in SeqIO.parse(fasta_file, "fasta"):
        homologous_seq = record.seq[35:55].reverse_complement()
        seq2sgRNA[homologous_seq] = record.id
        sgRNA2seq[record.id] = homologous_seq
    guides_list = list(sgRNA2seq.keys())
    return sgRNA2seq, seq2sgRNA, guides_list


def build_quality_check_dict(position_file: str):
    """
    Build a dictionary of crucial nucleotide positions for quality checking in filtering step
    """

    nt_sgRNA = {}
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


def qscore_filter(seq_qscores: list) -> bool:
    """
    Check if the quality scores of the crucial nucleotides are above the threshold
    """
    qthreshold = 0
    high_quality = True
    for base in seq_qscores:
        if (ord(base) - 33) <= qthreshold:
            high_quality = False
            break
    return high_quality


def seq_find(seq, qscore, left, right, target_len, hamming_dist):
    make_string = (
        "(" + left + "." * target_len + right + ")" + "{s<" + str(hamming_dist) + "}"
    )
    matches = re.finditer(make_string, str(seq))
    for i, match in enumerate(matches):
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


def check_qscore(nt_sgRNA, sgRNA1_name, qscore_sgRNA1, sgRNA2_name, qscore_sgRNA2):
    if sgRNA1_name in nt_sgRNA.keys():
        if sgRNA1_name in nt_sgRNA.keys():
            for nt1 in nt_sgRNA[sgRNA1_name]:
                a = qscore_filter(qscore_sgRNA1[int(nt1) - 1])
                if not a:
                    break
            for nt2 in nt_sgRNA[sgRNA2_name]:
                b = qscore_filter(qscore_sgRNA2[int(nt2) - 1])
                if not b:
                    break
    else:
        a = False
        b = False
    return a, b


def build_sgRNA_dicts(logger: logging.Logger):
    # function used for identifying the sgRNAs present in reads from HiSeq
    """
    Build dictionaries for forward and backward reads of the plasmid, BC sequences, and Q scores for each file
    """
    sgRNA1_dict = {}
    qscore_sgRNA1_dict = {}
    sgRNA2_dict = {}
    qscore_sgRNA2_dict = {}
    BC_dict = {}
    qscore_BC_dict = {}
    sequence = {}
    q_score = {}
    read_total = 0

    read_file = file_name

    logger.info(f"Processing file: {read_file}")
    read_1_file = f"{read_file}_R1_001.fastq"
    read_2_file = f"{read_file}_R2_001.fastq"

    logger.info(read_file)
    sp = read_file.split("/")
    file_id = sp[1]
    sgRNA1_dict[file_id] = []
    qscore_sgRNA1_dict[file_id] = []
    sgRNA2_dict[file_id] = []
    qscore_sgRNA2_dict[file_id] = []
    BC_dict[file_id] = []
    qscore_BC_dict[file_id] = []

    # one of these is the forward read and one is the backward read of an overlapping region
    with open(read_1_file) as fwdFile:
        while True:
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
            sgRNA2_dict[file_id].append(sgRNA2_seq)
            qscore_sgRNA2_dict[file_id].append(qscore_sgRNA2)
            BC_dict[file_id].append(BC_seq)
            qscore_BC_dict[file_id].append(qscore_BC)

    with open(read_2_file) as fwdFile:
        while True:
            next_n_lines = list(islice(fwdFile, 4))
            if not next_n_lines:
                break
            sequence = Seq(next_n_lines[1]).reverse_complement()
            q_score = next_n_lines[3][::-1]
            read_total += 1
            sgRNA1_seq, qscore_sgRNA1 = seq_find(
                sequence, q_score, "CTAGCTCTAAAAC", "A", 20, 3
            )
            sgRNA1_dict[file_id].append(sgRNA1_seq)
            qscore_sgRNA1_dict[file_id].append(qscore_sgRNA1)

    return (
        sgRNA1_dict,
        qscore_sgRNA1_dict,
        sgRNA2_dict,
        qscore_sgRNA2_dict,
        BC_dict,
        qscore_BC_dict,
        read_total,
    )


def get_overall_counts(sgRNA1_dict, sgRNA2_dict, BC_dict, logger: logging.Logger):
    """
    Build overall counts of reads with sgRNA1, sgRNA2, and BC found for each file

    This is just for logging purposes.
    """
    all_seqs_total = 0
    reads_dict = {}
    read_file = file_name

    sp = read_file.split("/")
    file_id = sp[1]
    reads_dict[file_id] = []

    sgRNA1_sum = sum([x != "Not found" for x in sgRNA1_dict[file_id]])
    sgRNA2_sum = sum([x != "Not found" for x in sgRNA2_dict[file_id]])
    BC_sum = sum([x != "Not found" for x in BC_dict[file_id]])
    total = len(sgRNA1_dict[file_id])
    reads_dict[file_id].append(total)

    logger.info(f"sgRNA1 found: {sgRNA1_sum}")
    logger.info(f"sgRNA2 found: {sgRNA2_sum}")
    logger.info(f"BC found: {BC_sum}")
    logger.info(f"Total reads: {total}")

    both = [
        len(x) == 20 and len(y) == 20 and len(z) == 6
        for x, y, z in zip(sgRNA1_dict[file_id], sgRNA2_dict[file_id], BC_dict[file_id])
    ]
    reads_dict[file_id].append(sum(both))
    Success_print = "Success: " + str(sum(both))
    logger.info(Success_print)
    print("")

    all_seqs_total += sum(both)

    return reads_dict, all_seqs_total


def build_sgRNA_count_dicts(
    guides_list,
    sgRNA1_dict,
    qscore_sgRNA1_dict,
    sgRNA2_dict,
    qscore_sgRNA2_dict,
    BC_dict,
    nt_sgRNA,
    seq2sgRNA,
    qthreshold,
    logger,
):
    """
    Build separate counts for each barcode
    """
    ab = 0
    onlya = 0
    onlyb = 0
    none = 0
    failed = 0

    read_file = file_name

    sp = read_file.split("/")
    file_id = sp[1]
    pairwise_lib[file_id] = {}

    ## seeds the pairwise_lib dictionaries
    for BC in ["TGAAAG", "CATGAT", "CCATGC", "TCATAC", "TAGACT", "ACTAGG", "Other"]:
        pairwise_lib[file_id][BC] = pd.DataFrame(
            np.zeros([len(guides_list), len(guides_list)]),
            guides_list,  # ty: ignore
            guides_list,  # ty: ignore
        )

    ## populates the pairwise_lib dictionaries
    for i, guide1 in enumerate(sgRNA1_dict[file_id]):
        guide2 = sgRNA2_dict[file_id][i]
        BC = BC_dict[file_id][i]
        if len(guide1) == 20 and len(guide2) == 20:
            if BC in ["TGAAAG", "CATGAT", "CCATGC", "TCATAC", "TAGACT", "ACTAGG"]:
                try:
                    guide1_id = seq2sgRNA[guide1]
                    guide2_id = seq2sgRNA[guide2]

                    a, b = check_qscore(
                        nt_sgRNA,
                        guide1_id,
                        qscore_sgRNA1_dict[file_id][i],
                        guide2_id,
                        qscore_sgRNA2_dict[file_id][i],
                    )
                    if a and b:
                        pairwise_lib[file_id][BC].at[guide1_id, guide2_id] += 1
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

                    pairwise_lib[file_id]["Other"].at[guide1_id, guide2_id] += 1

                except (KeyError, IndexError, UnboundLocalError):
                    failed += 1
    sum_low = onlya + onlyb + none
    sum_all = sum_low + ab

    logger.info("timepoint: " + str(file_id))
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


def reindex_pairwise_dict(guides_list):
    read_file = file_name
    sp = read_file.split("/")
    file_id = sp[1]

    for BC in pairwise_lib[file_id].keys():
        guides_list.sort()
        pairwise_lib[file_id][BC] = pairwise_lib[file_id][BC][guides_list].T
        pairwise_lib[file_id][BC] = pairwise_lib[file_id][BC][guides_list].T


def save_unfiltered_counts(filepath: str = "BarcodedCounts/unfiltered/"):
    date = datetime.date.today()
    for TP in pairwise_lib.keys():
        for BC in pairwise_lib[TP].keys():
            pd.DataFrame(pairwise_lib[TP][BC]).to_csv(
                filepath + str(date) + "_" + TP + "_" + BC + "_counts.csv"
            )


if __name__ == "__main__":
    logger = get_logger("counting_code.log")

    # output dictionary
    pairwise_lib = {}
    BC_list = ["TGAAAG", "CATGAT", "CCATGC", "TCATAC", "TAGACT", "ACTAGG"]

    # read through a single file name
    file_name = open(sys.argv[1]).readline().strip()

    logger.info(f"Currently processing: {file_name}")

    sgRNA2seq, seq2sgRNA, guides_list = build_guide_dict("PW_Folate_corrected.fa")
    nt_sgRNA = build_quality_check_dict("nt_sgRNA_FolateSinglesCorrected.txt")
    sgRNA1, qscore_sgRNA1, sgRNA2, qscore_sgRNA2, BC, qscore_BC, read_total = (
        build_sgRNA_dicts(logger)
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
        0,
        logger,
    )
    reindex_pairwise_dict(guides_list)
    save_unfiltered_counts()
