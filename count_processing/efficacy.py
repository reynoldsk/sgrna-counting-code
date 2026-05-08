import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from Bio import SeqIO
from scipy.stats import linregress
import matplotlib.pyplot as plt
import logging
from .experiment import Experiment
from pathlib import Path

"""
We attempt to calculate repression efficacy of sgRNAs
without using the gold standard method (qtPCR) and instead
we need to make a certain number of assumptions to create a ranking 
of sgRNA efficacy as a proxy for the true repression efficacy.

We have a few options:

1. assume that the growth rate is monotonically related to 
repression efficacy, and sort the sgRNAs by growth rate, then 
using that ranking to build the epistatis model. This makes 
a questionable assumption that growth rate is related to efficacy.

From Phil:

"The final repression efficiency strategy I created and used calculates for each media condition the range of growth rates observed across all guides targeting a single gene. The ranks are weighted (multiplied) by this calculated range of values for each media condition at the stage where the ranks are averaged across media conditions. Then rescaled between 0 and 1 as described above."

2. assume that we can rank the sgRNA efficacy by the position and 
number of mismatches. Generally, sgRNAs targeting earlier in the 
gene will be more effective, and sgRNAs with fewer mismatches will 
be more effective. We can use this ranking to build the epistasis model. 
This makes the assumption that the number and position of mismatches
is related to efficacy. 

3. Machine learning lmfao
"""


class EfficacyEstimator:
    """
    Estimate repression efficacy of a set of sgRNAs for a
    specific vial and set of genes using one of several
    methods where we do not have the gold standard qtPCR data to directly measure repression efficacy.
    """

    def __init__(self, average_growth_rates: pd.DataFrame, genes: set[str] | Path):
        if isinstance(genes, Path):
            with open(genes, "r") as f:
                self.genes = [line.strip() for line in f]
                self.genes = set(self.genes)  # Convert to set for uniqueness
        else:
            self.genes = genes
        self.average_growth_rates = average_growth_rates

    def rank_sgRNAs_by_mismatch_position(self) -> pd.DataFrame:
        """
        Rank sgRNAs by their mismatch position and number of mismatches, assuming that sgRNAs with fewer mismatches and mismatches further from the PAM are more effective.

        We rank first based on position of the mismatch,
        and then by the number of mismatches.

        Returns
        -------
        pd.DataFrame
            DataFrame with sgRNAs ranked by mismatch position and number, including columns for sgRNA identifier, number of mismatches, and mismatch positions.
        """

        pass

    def rank_sgRNAs_by_growth_rate(self) -> pd.DataFrame:
        """
        Rank sgRNAs by their growth rates, assuming that lower growth rates correspond to higher repression efficacy.

        Rank, then normalize the growth rates to a 0-1 scale for
        each vial, weigh the ranking by this normalized growth
        rate, and then normalize again to 0-1 to get
        the repression efficacy, per Phil

        Returns
        -------
        pd.DataFrame
            DataFrame with sgRNAs ranked by growth rate, including columns for sgRNA identifier, growth rate, and gene.
        """

        ranked_sgRNAs = self.average_growth_rates.sort_values(by="mean", axis=1)

        genes_in_data = [
            next((g for g in self.genes if g in sgRNA), None)
            for sgRNA in ranked_sgRNAs.columns
        ]
        ranked_sgRNAs.columns = pd.MultiIndex.from_arrays(
            [ranked_sgRNAs.columns, genes_in_data], names=["sgRNA", "gene"]
        )
        
        return ranked_sgRNAs
