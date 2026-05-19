import pandas as pd
import numpy as np
from pathlib import Path
import logging

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

    def __init__(
        self,
        average_growth_rates: pd.DataFrame,
        genes: set[str] | Path,
        sgRNAs: set[str],
    ):
        self.sgRNAs = sgRNAs
        if isinstance(genes, Path):
            with open(genes, "r") as f:
                self.genes = [line.strip() for line in f]
                self.genes = set(self.genes)  # Convert to set for uniqueness
        else:
            self.genes = genes
        self.average_growth_rates = average_growth_rates

    def rank_sgRNAs_by_mismatch_position(
        self, negative_control: str | None = None
    ) -> pd.DataFrame:
        """
        Rank sgRNAs by their mismatch position and number of mismatches, assuming that sgRNAs with fewer mismatches and mismatches further from the PAM are more effective.

        We rank first based on position of the mismatch,
        and then by the number of mismatches.

        We assume a certain naming convention for each sgRNA:

        <gene name>_<attempt # at making an sgRNA>_<nucleotide in coding sequence targeted>_<Mismatch Strategy>

        Mismatch strategy:
        - C: no mismatches
        - B/W: some mismatches, followed by MM (number of mismatches))
        if B or W

        Example:
        fmt_1_37_B_MM4
        - fmt gene
        - 1st attempt
        - targets 37th nucleotide
        - B mismatch strategy
        - 4 mismatches

        Sometimes they don't have a mismatch strategy, in which case we need to process num mismatches directly

        This does not depend on growth rates at all.

        Parameters
        ----------
        negative_control : str | None
            Identifier for negative control sgRNA, if it does not follow the same naming convention.
            If provided, will be included in the ranking manually

        Returns
        -------
        pd.DataFrame
            DataFrame with sgRNAs ranked by mismatch position and number, including columns for sgRNA identifier,
            number of mismatches, and mismatch positions.
        """
        sgRNA_rankings = []

        ## get mismatch position and number of mismatches from sgRNA naming
        for sgRNA in self.sgRNAs:
            parts = sgRNA.split("_")

            ## negative controls may not follow the same convention
            if len(parts) < 4 or len(parts) > 5:
                logging.warning(
                    f"sgRNA {sgRNA} does not follow expected naming convention."
                )
                continue
            gene, attempt, position_str, mismatch_strategy = parts[:4]
            try:
                position = int(position_str)
            except ValueError:
                logging.warning(f"sgRNA {sgRNA} has invalid position: {position_str}")
                continue
            num_mismatches = (
                0
                if mismatch_strategy == "C"
                else int(parts[4].split("MM")[1])
                if len(parts) == 5
                else int(mismatch_strategy.split("MM")[1])
            )
            sgRNA_rankings.append((sgRNA, gene, position, num_mismatches))

        ## add the negative control if one is provided
        if negative_control:
            sgRNA_rankings.append(
                (
                    negative_control,
                    negative_control.split("_")[0],
                    float("inf"),
                    float("inf"),
                )
            )  # Add negative control at the end of the ranking

        ranked_sgRNAs = pd.DataFrame(
            sgRNA_rankings,
            columns=np.array(["sgRNA", "gene", "position", "num_mismatches"]),
        )

        ranked_sgRNAs = ranked_sgRNAs.sort_values(
            by=["position", "num_mismatches"], ascending=[True, True]
        ).reset_index(drop=True)

        growth_rates = self.average_growth_rates.copy()

        genes_in_data = [
            next((g for g in self.genes if g in sgRNA), None)
            for sgRNA in ranked_sgRNAs["sgRNA"]
        ]

        growth_rates.columns = pd.MultiIndex.from_arrays(
            [ranked_sgRNAs["sgRNA"], genes_in_data], names=["sgRNA", "gene"]
        )

        return growth_rates

    def rank_sgRNAs_by_growth_rate(self) -> pd.DataFrame:
        """
        Rank sgRNAs by their growth rates, assuming that lower growth rates correspond to higher repression efficacy.

        This method depends on cross-vial ranking.

        Returns
        -------
        pd.DataFrame
            DataFrame with sgRNAs ranked by growth rate, including columns for sgRNA identifier, growth rate, and gene.
        """

        ranked_sgRNAs = self.average_growth_rates.copy()

        ranked_sgRNAs = ranked_sgRNAs.sort_values(by="mean", axis=1, ascending=True)

        ranked_cols = (
            ranked_sgRNAs.columns.get_level_values("sgRNA")
            if isinstance(ranked_sgRNAs.columns, pd.MultiIndex)
            else ranked_sgRNAs.columns
        )

        genes_in_data = [
            next((g for g in self.genes if g in sgRNA), None) for sgRNA in ranked_cols
        ]
        ranked_sgRNAs.columns = pd.MultiIndex.from_arrays(
            [ranked_cols, genes_in_data], names=["sgRNA", "gene"]
        )

        return ranked_sgRNAs

    def get_gene_epistasis_dict(
        self, ranked_sgrnas: pd.DataFrame, gene: str
    ) -> dict[str, tuple[float, float, pd.Series]]:
        """
        Compute repression efficacy for sgRNAs targeting a specific gene.

        Groups ranked_sgrnas by the gene level of the MultiIndex columns, then
        applies per Phil:
        1. Rank sgRNAs within the gene by mean value (ascending: lower = rank 1)
        2. Weight rank by normalized value

        Works with output from either rank_sgRNAs_by_growth_rate or
        rank_sgRNAs_by_mismatch_position.

        rank by growth_rate requires another step across media condition:

        3. Average rank across media conditions
        4. Rescale to [0, 1]

        Parameters
        ----------
        ranked_sgrnas : pd.DataFrame
            Index is ["mean", "std", "sem"]; columns are a MultiIndex (sgRNA, gene).
        gene : str
            Gene to compute efficacy for.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping of sgRNA identifier to (repression efficacy score in [0, 1], mean value).
        """

        gene_data = ranked_sgrnas.xs(gene, level="gene", axis=1)
        mean_vals = gene_data.loc["mean"]
        abs_vals = mean_vals.abs()

        ranks = abs_vals.rank(ascending=True)

        # Y-axis: signed growth rate normalized to [-1, 1], preserving sign through 0
        # max_abs = abs_vals.max()
        # normalized = (
        #     mean_vals / max_abs
        #     if max_abs != 0
        #     else pd.Series(0.0, index=mean_vals.index)
        # )

        # looking at growth rate effect size:
        normalized = (abs_vals - abs_vals.min()) / (abs_vals.max() - abs_vals.min())

        # X-axis ordering: rank by absolute effect, weighted by normalized absolute range
        val_range = abs_vals.max() - abs_vals.min()
        if val_range == 0:
            return {
                sgRNA: (0.0, mean_vals[sgRNA], normalized[sgRNA])
                for sgRNA in gene_data.columns
            }
        abs_normalized = (abs_vals - abs_vals.min()) / val_range
        weighted = ranks * abs_normalized

        span = weighted.max() - weighted.min()
        if span == 0:
            return {
                sgRNA: (0.0, mean_vals[sgRNA], normalized[sgRNA])
                for sgRNA in gene_data.columns
            }

        efficacy = (weighted - weighted.min()) / span
        return {
            sgRNA: (efficacy[sgRNA], mean_vals[sgRNA], normalized[sgRNA])
            for sgRNA in gene_data.columns
        }

    def average_ranks_across_conditions(
        self,
        ranked_sgrnas_list: list[pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        """
        Average per-gene weighted ranks across media conditions and rescale to [0, 1].

        Per Phil's method: for each condition, rank sgRNAs within a gene by absolute
        mean growth rate, then weight each rank by the gene-level range of growth rates.
        Average the weighted ranks across conditions, then rescale per gene to [0, 1].

        Parameters
        ----------
        ranked_sgrnas_list : list[pd.DataFrame]
            One DataFrame per media condition (vial). Each has MultiIndex columns
            (sgRNA, gene) and index containing at least "mean".

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping of gene -> {sgRNA -> efficacy in [0, 1]}.
        """
        # Accumulate weighted ranks per sgRNA across conditions
        all_weighted: dict[str, list[float]] = {}
        sgRNA_to_gene: dict[str, str] = {}

        for ranked_df in ranked_sgrnas_list:
            mean_vals = ranked_df.loc["mean"]
            for sgRNA, gene in zip(
                mean_vals.index.get_level_values("sgRNA"),
                mean_vals.index.get_level_values("gene"),
            ):
                sgRNA_to_gene[sgRNA] = gene

            for gene in mean_vals.index.get_level_values("gene").unique():
                gene_means = mean_vals.xs(gene, level="gene").abs()
                val_range = gene_means.max() - gene_means.min()
                ranks = gene_means.rank(ascending=True)
                weighted = ranks * val_range
                for sgRNA, w in weighted.items():
                    all_weighted.setdefault(sgRNA, []).append(w)

        avg_weighted = pd.Series(
            {sgRNA: np.mean(ws) for sgRNA, ws in all_weighted.items()}
        )

        # Group by gene and rescale each gene's sgRNAs to [0, 1]
        result: dict[str, dict[str, float]] = {}
        for gene in set(sgRNA_to_gene.values()):
            sgrnas = [s for s, g in sgRNA_to_gene.items() if g == gene]
            gene_vals = avg_weighted.reindex(sgrnas)
            span = gene_vals.max() - gene_vals.min()
            if span == 0:
                result[gene] = {s: 0.0 for s in sgrnas}
            else:
                rescaled = (gene_vals - gene_vals.min()) / span
                result[gene] = rescaled.to_dict()
        

        return result
