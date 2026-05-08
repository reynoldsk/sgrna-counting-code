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

"""
Experiment should define an experiment and its assumptions,
like paired reads or single reads via extending the 
Experiment class and implementing the aggregate_sgrna method.

Experiment also defines timepoints and barcodes, 
which are used to split the counts by time and by barcode.

An experiment contains all of the reads for a single 
experimental condition (vial), and runs the growth rate
calculation for each barcode. 

Barcodes are biological replicates. Multiple runs with 
different biological constructs.

Each experiment should be able to run the whole pipeline:
load counts --> split by time and barcode --> 
run each barcode through growth rate calculation 
--> average growth rates across barcodes
"""


class Experiment(ABC):
    """
    Initialize an experiment with its assumptions and conditions.

    Fields:
    ---------
    - counts_directory: The directory containing count files for the experiment.
    - vial_id: The ID of the vial for which to load counts.
    - timepoints: A list of time points for the experiment.
    - files: a list of files in the counts directory that match the vial_id pattern.
    - barcodes: A list of barcodes for the experiment, which are technical replicates.
    - sgRNA2seq: A dictionary mapping sgRNA ids to their corresponding homology sequences.
    - seq2sgRNA: A dictionary mapping homology sequences to their corresponding sgRNA ids
    - guides_list: A list of sgRNA ids.
    - plotting: Whether to generate plots for the experiment.
    - plot_directory: The directory where plots will be saved if plotting is True.
    - logger: A logging.Logger instance for logging messages, or None to print to console.
    - neg_control_sgRNA: The ID of the negative control sgRNA used for normalization.
    - intercepts: A DataFrame to store intercepts from growth rate fitting for plotting purposes.
    """

    def __init__(
        self,
        counts_directory: Path,
        vial_id: str,
        timepoints: list[str],
        barcodes: list[str],
        sgRNAs: tuple[dict[str, str], dict[str, str], list[str]] | Path,
        neg_control_sgRNA: str = "NTC",
        plotting: bool = True,
        plot_directory: Path = Path("plots-monitoring"),
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the experiment with the directory containing count files.

        Parameters:
        counts_directory (Path): The path to the directory containing count files.
        vial_id (str): The ID of the vial for which to load counts.
        timepoints (list[str]): A list of time points for the experiment.
        barcodes (list[str]): A list of barcodes for the experiment, which are biological replicates.
        sgRNAs (tuple[dict[str, str], dict[str, str], list[str]] | Path): Either a path to a fasta file
        containing sgRNA sequences and ids, or a tuple of (sgRNA2seq, seq2sgRNA, guides_list) where:
            - sgRNA2seq: A dictionary mapping sgRNA ids to their corresponding homology sequences.
            - seq2sgRNA: A dictionary mapping homology sequences to their corresponding sgRNA ids.
            - guides_list: A list of sgRNA ids.
        plotting (bool): Whether to generate plots for the experiment. Default is True.
        neg_control_sgRNA (str): The ID of the negative control sgRNA. Default is "NTC".
        """
        self.counts_directory = counts_directory
        self.vial_id = vial_id
        self.files = list(counts_directory.rglob(f"*{vial_id}*.csv"))
        self.timepoints = timepoints
        self.barcodes = barcodes
        if isinstance(sgRNAs, Path):
            self.build_guide_dict(sgRNAs)
        else:
            self.sgRNA2seq, self.seq2sgRNA, self.guides_list = sgRNAs
        self.plotting = plotting
        self.plot_directory = plot_directory
        os.makedirs(self.plot_directory, exist_ok=True)
        self.logger = logger
        self.neg_control_sgRNA = neg_control_sgRNA

    def build_guide_dict(
        self,
        fasta_file: Path,
    ) -> None:
        """
        From the fasta file, take the sequences and their id to create a dictionary of sgRNA id to
        homology sequence and homology sequence to sgRNA id, as well as a list of the sgRNA ids.

        Parameters
        -----------
        fasta_file : Path
            The path to the fasta file containing the sgRNA sequences and their ids.

        Returns
        --------
        None
            This method does not return a value, but it populates the instance variables:
            - self.sgRNA2seq: A dictionary mapping sgRNA ids to their corresponding homology sequences.
            - self.seq2sgRNA: A dictionary mapping homology sequences to their corresponding sgRNA ids.
            - self.guides_list: A list of sgRNA ids.
        """

        sgRNA2seq: dict[str, str] = {}
        seq2sgRNA: dict[str, str] = {}

        # find homology + reverse complement of homology from PW_Folate_corrected.fa
        for record in SeqIO.parse(fasta_file, "fasta"):
            homologous_seq = str(record.seq[35:55].reverse_complement())
            seq2sgRNA[homologous_seq] = str(record.id)
            sgRNA2seq[str(record.id)] = homologous_seq
        guides_list = list(sgRNA2seq.keys())
        self.sgRNA2seq = sgRNA2seq
        self.seq2sgRNA = seq2sgRNA
        self.guides_list = guides_list

    def read_time_library(self) -> pd.DataFrame:
        """
        Read all files matching this experiment's vial_id and build a MultiIndex DataFrame.

        Returns a DataFrame with a (timepoint, barcode) MultiIndex and sgRNAs as columns,
        where each row is the aggregated counts for that timepoint/barcode combination.
        """
        records = []
        for file in self.files:
            timepoint = next((tp for tp in self.timepoints if tp in file.name), None)
            barcode = next((bc for bc in self.barcodes if bc in file.name), None)
            if timepoint is None or barcode is None:
                continue
            counts_df = pd.read_csv(file, index_col=0)
            aggregated = self.aggregate_sgrna(counts_df)
            records.append((timepoint, barcode, aggregated))

        index = pd.MultiIndex.from_tuples(
            [(tp, bc) for tp, bc, _ in records],
            names=["timepoint", "barcode"],
        )
        return pd.DataFrame(
            [series for _, _, series in records],
            index=index,
        ).sort_index(level="timepoint")

    @contextmanager
    def _figure(self, figsize: tuple[int, int] = (8, 6)):
        """
        Context manager for creating and closing a matplotlib figure. Allows us to plot a set of figures on the
        same figure axes.
        """
        fig, ax = plt.subplots(figsize=figsize)
        try:
            yield fig, ax
        finally:
            plt.close(fig)

    def relative_frequency(self, counts: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative frequency of each sgRNA by dividing by the normalizing counts.

        Parameters
        ----------
        counts : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs with raw counts.

        Returns
        -------
        pd.DataFrame
            Same shape as counts; each value is divided by the experiment-specific normalizer.
        """
        normalizing = self.get_normalizing_counts(counts)
        return counts.div(normalizing, axis=0)

    def normalize_to_t0(self, freq: pd.DataFrame, t0: str) -> pd.DataFrame:
        """
        Normalize relative frequencies against the same sgRNA's frequency at t0,
        per barcode.

        Parameters
        ----------
        freq : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs of relative frequencies.
        t0 : str
            The timepoint label to use as the baseline (e.g. "T00").

        Returns
        -------
        pd.DataFrame
            freq divided by t0 values; t0 rows will all be 1.0.
        """
        t0_freq = freq.xs(t0, level="timepoint")
        return freq.div(t0_freq, level="barcode")

    def log_transform(self, freq: pd.DataFrame) -> pd.DataFrame:
        """
        Log2-transform a relative frequency DataFrame.

        Parameters
        ----------
        freq : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs.

        Returns
        -------
        pd.DataFrame
            Log2 of freq; t0-normalized input will have 0 at baseline.
        """
        return np.log2(freq)  # ty: ignore ; this does return a DataFrame actually

    def growth_rates(
        self, log_freq: pd.DataFrame, timepoint_values: dict[str, int]
    ) -> pd.DataFrame:
        """
        Compute per-sgRNA growth rates as the slope of log2 frequency vs time,
        fit separately for each barcode.

        Parameters
        ----------
        log_freq : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs of log2 frequencies.
        timepoint_values : dict[str, int]
            Mapping from timepoint label to numeric value (e.g. {"T-3": -3, "T00": 0, "T08": 8}).

        Returns
        -------
        pd.DataFrame
            DataFrame with barcode as index and sgRNAs as columns containing growth rate slopes.
        """

        def _col_mask(col: pd.Series, t_labels: list[str]) -> float:
            mask = np.isfinite(col.values)
            if mask.sum() < 2:
                return np.nan
            t = np.array([timepoint_values[tp] for tp in t_labels[mask]])
            return linregress(t, col.values[mask]).slope

        def _slope(group: pd.DataFrame) -> pd.Series:
            t_labels = group.index.get_level_values("timepoint")
            return group.apply(_col_mask, args=(t_labels,))

        def _intercept(group: pd.DataFrame) -> pd.Series:
            t_labels = group.index.get_level_values("timepoint")
            return group.apply(_col_mask, args=(t_labels,))

        grs = log_freq.groupby(level="barcode").apply(_slope)

        ## save intercepts in the object for plotting
        self.intercepts = log_freq.groupby(level="barcode").apply(_intercept)

        if self.plotting:
            sgRNA = self.guides_list[0]  # Just plot the first sgRNA for now
            self.plot_growth_rates(sgRNA, grs, log_freq, timepoint_values)

        return grs

    def plot_growth_rates(
        self,
        sgRNA: str,
        grs: pd.DataFrame,
        log_freq: pd.DataFrame,
        timepoint_values: dict[str, int],
    ) -> None:
        """
        Plot the growth of a single sgRNA over time by barcode, with points for the log2 
        relative frequencies and lines for the fitted growth rates.

        Parameters
        ----------
        sgRNA : str
            The sgRNA to plot.
        grs : pd.DataFrame
            DataFrame with barcode as index and sgRNAs as columns containing growth rate slopes.
        log_freq : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs of log2 frequencies.
        timepoint_values : dict[str, int]
            Mapping from timepoint label to numeric value (e.g. {"T-3": -3, "T00": 0, ... "T08": 8}).
        """

        with self._figure() as (fig, ax):
            for barcode in grs.index:
                group = log_freq.xs(barcode, level="barcode")
                t = np.array(
                    [
                        timepoint_values[tp]
                        for tp in group.index.get_level_values("timepoint")
                    ]
                )
                ax.scatter(t, group[sgRNA], marker="o", label=f"Barcode {barcode}")
                ax.plot(
                    t,
                    grs.loc[barcode, sgRNA] * t + self.intercepts.loc[barcode, sgRNA],
                    marker="x",
                    label=f"Fit {barcode}",
                )
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel(f"Log2 Relative Frequency of {sgRNA}")
            ax.set_title(f"Growth of {sgRNA} over time by barcode")
            ax.legend()
            ax.grid(True)
            fig.savefig(f"{self.plot_directory}/growth_plot_{self.vial_id}_{sgRNA}.png")

    def escaper_correction(self, growth_rates: pd.DataFrame) -> pd.DataFrame:
        """
        Apply escaper correction to the growth rates.

        Parameters
        ----------
        growth_rates : pd.DataFrame
            DataFrame with barcode as index and sgRNAs as columns containing growth rate slopes.

        Returns
        -------
        pd.DataFrame
            Corrected growth rates with outlier (barcode, sgRNA) cells set to NaN.
        """
        corrected = growth_rates.copy()
        results = growth_rates.apply(self.dixon_q_test, axis=0)
        # results rows: 0 = q_statistic, 1 = outlier_barcode (or None)
        outlier_barcodes = results.iloc[1]

        for sgRNA, barcode in outlier_barcodes.items():
            if barcode is not None and not pd.isna(barcode):
                self.log(
                    f"Outlier detected for {sgRNA} in barcode {barcode}. Applying escaper correction.",
                    logging.INFO,
                )
                corrected.loc[barcode, sgRNA] = np.nan

        return corrected

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a message using the experiment's logger if available, otherwise print to console.

        Parameters
        ----------
        message : str
            The message to log.
        level : int
            The logging level (e.g. logging.INFO, logging.WARNING). Default is logging.INFO.

        Returns
        -------
        None
            This method does not return a value, but logs the message using the logger or prints it to the console.
        """
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message)

    @classmethod
    def dixon_critical_value(cls, n: int) -> float | None:
        """
        Get the critical value for Dixon's Q test based on the number of observations.

        Parameters
        ----------
        n : int
            The number of observations in the dataset.

        Returns
        -------
        float
            The critical value for Dixon's Q test at a 95% confidence level. 
            If n is not in the predefined range, returns None.
        """
        q_critical: dict[int, float] = {
            3: 0.970,
            4: 0.829,
            5: 0.710,
            6: 0.625,
            7: 0.568,
            8: 0.526,
            9: 0.493,
            10: 0.466,
            # Add more values as needed
        }
        return q_critical.get(n, None)  # Return None if n is not in the dictionary

    @classmethod
    def dixon_q_test(cls, data: pd.Series) -> tuple[float, str | None]:
        """
        Perform Dixon's Q test for outliers on a Series of growth rates.

        Gets the minimum outlier or the maximum outlier, if there is any

        Parameters
        ----------
        data : pd.Series
            Series of growth rates for a single sgRNA across barcodes.

        Returns
        -------
        tuple[float, str | None]
            The test statistic and the barcode that is an outlier.
        """

        sorted_data = data.sort_values()
        n = len(sorted_data)
        if n < 3:
            return False, None  # Not enough data points for the test

        ## 95% confidence level interval

        ## if the 2 datapoints are the same, can't do the test
        if sorted_data.iloc[0] == sorted_data.iloc[-1]:
            return False, None

        ## check min and max q statistics:
        min_q_statistic = (sorted_data.iloc[1] - sorted_data.iloc[0]) / (
            sorted_data.iloc[-1] - sorted_data.iloc[0]
        )
        max_q_statistic = (sorted_data.iloc[-1] - sorted_data.iloc[-2]) / (
            sorted_data.iloc[-1] - sorted_data.iloc[0]
        )
        if min_q_statistic > max_q_statistic:
            q_statistic = min_q_statistic
            index_to_remove = 0
        elif max_q_statistic > min_q_statistic:
            q_statistic = max_q_statistic
            index_to_remove = -1
        else:
            ## this case should never happen
            return False, None  # Both statistics are the same, can't determine outlier

        q_critical = cls.dixon_critical_value(n)
        is_outlier = q_statistic > q_critical if q_critical is not None else False

        outlier_barcode = sorted_data.index[index_to_remove] if is_outlier else None
        return q_statistic, outlier_barcode

    @abstractmethod
    def get_normalizing_counts(self, counts: pd.DataFrame) -> pd.Series:
        """
        Extract the normalizing counts from the counts DataFrame — typically the counts of a
        non-targeting control sgRNA. Returns a Series indexed by (timepoint, barcode) used
        to divide each row in relative_frequency.

        Parameters
        ----------
        counts: pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs with raw counts.

        Returns
        -------
        pd.Series
            Series indexed by (timepoint, barcode) containing the normalizing counts for each row in counts
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def aggregate_sgrna(self, counts_df: pd.DataFrame) -> pd.Series:
        """
        Collapse the pairwise sgRNA count matrix into a 1D Series (sgRNA → count)
        according to the experimental design.

        Parameters
        ----------
        counts_df : pd.DataFrame
            DataFrame with sgRNAs as both rows and columns

        Returns
        -------
        pd.Series
            Series indexed by sgRNA with the aggregated counts according to the experiment design.
        """
        raise NotImplementedError("Subclasses must implement this method")


class SelfSelfExperiment(Experiment):
    """
    Plasmid has the same sgRNAs in both directions.
    """

    def aggregate_sgrna(self, counts_df: pd.DataFrame) -> pd.Series:
        """
        For this experiment design, the count files have sgRNAs in both rows and columns.
        We want to get the diagonals.

        Parameters
        ----------
        counts_df : pd.DataFrame
            DataFrame with sgRNAs as both rows and columns.

        Returns
        -------
        pd.Series
            Series indexed by sgRNA with the sum of counts across columns.
        """
        sgRNA_counts = []
        for sgRNA in counts_df.columns:
            sgRNA_counts.append(counts_df.loc[sgRNA, sgRNA])
        return pd.Series(sgRNA_counts, index=counts_df.columns)

    def get_normalizing_counts(self, counts: pd.DataFrame) -> pd.Series:
        """
        For this experiment design, the normalizing counts are the
        counts of the non-targeting control sgRNA, which is the same in rows and columns.

        Parameters
        ----------
        counts : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs with raw counts.

        Returns
        -------
        pd.Series
            Series indexed by (timepoint, barcode) containing the counts of the non-targeting control sgRNA.
        """
        return counts[self.neg_control_sgRNA]


class SelfNonTargetingExperiment(Experiment):
    """
    Plasmid has the same sgRNAs in one direction, and non-targeting controls in the other.
    """

    def aggregate_sgrna(self, counts_df: pd.DataFrame) -> pd.Series:
        """
        For this experiment design, the count files have sgRNAs in rows and non-targeting controls in columns.
        We want to sum across the non-targeting control columns to get a single count per sgRNA.

        Parameters
        ----------
        counts_df : pd.DataFrame
            DataFrame with sgRNAs as rows and non-targeting controls as columns.

        Returns
        -------
        pd.Series
            Series indexed by sgRNA with the sum of counts across non-targeting controls.
        """
        return counts_df[self.neg_control_sgRNA]

    def get_normalizing_counts(self, counts: pd.DataFrame) -> pd.Series:
        """
        For this experiment design, the normalizing counts are the
        counts of the non-targeting control sgRNA.

        Parameters
        ----------
        counts : pd.DataFrame
            MultiIndex DataFrame (timepoint, barcode) × sgRNAs with raw counts.

        Returns
        -------
        pd.Series
            Series indexed by (timepoint, barcode) containing the counts of the non-targeting control sgRNA.
        """
        return counts[self.neg_control_sgRNA]