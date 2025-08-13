"""Computing mutation rates"""

import pandas as pd
import numpy as np
from modules import glm


# Nucleotide alphabet and mutations
letters = ["A", "C", "G", "T"]
mut_types = [b1+b2 for b1 in letters for b2 in letters if b1 != b2]


class Rates:
    # Initialize all rates to zeros
    def __init__(self,):
        unpaired = [0]
        # light_switch = [l_min, l_max]
        light_switch = [False]

        unique_muts = []

        for m in mut_types:
            for x_l in letters:
                for x_r in letters:
                    for p in unpaired:
                        if m in ["CT", "AT", "CG", "GC"]:
                            for l in light_switch:
                                unique_muts.append([m, x_l + m[0] + x_r, p, l, 0.0])
                        else:
                            unique_muts.append([m, x_l + m[0] + x_r, p, False, 0.0])

        self.rates = pd.DataFrame(
            unique_muts,
            columns=[
                "mut_type",
                "motif",
                "unpaired",
                "nt_site_before_boundary",
                "rate",
            ],
        )

    # Populate rates according to inferred GLM
    def populate_rates(self, count_df):
        # Infer GLM
        general_linear_model = glm.GeneralLinearModel(
            # ["global_context", "rna_structure", "local_context"]
            ["local_context"]
        )
        general_linear_model.train(count_df)
        s_max = count_df.nt_site.max()
        rates = self.rates
        rates["nt_site"] = rates["nt_site_before_boundary"].apply(
            lambda x: s_max
        )

        # Populate rates according to GLM
        for m in mut_types:
            rates.loc[rates["mut_type"] == m, "rate"] = (
                np.exp(
                    general_linear_model.create_data_matrix(
                        rates[rates["mut_type"] == m], m
                    )
                    @ general_linear_model.W[m]
                )
                - 0.5
            )

        rates["predicted_count"] = rates["rate"]
        rates.drop(columns=["nt_site"], inplace=True)

        # Rescale counts by total number of mutations and number of synonymous sites
        tot_mut = count_df.actual_count.sum()
        n_ss = count_df.shape[0]

        rates.rate *= n_ss / tot_mut

        # Add column `condition` with tuple summary of mutation conditions
        # cond_cols = ["mut_type", "motif", "unpaired", "nt_site_before_boundary"]
        cond_cols = ["mut_type", "motif"]
        rates["condition"] = rates[cond_cols].apply(tuple, axis=1)

    def predicted_counts(self, count_df_syn):
        T = self.evol_time(count_df_syn)
        gen_comp = self.genome_composition(count_df_syn)

        assert self.rates.shape[0] == len(gen_comp)

        new_cols = {"predicted_count": (T * self.rates.rate), "cond_count": gen_comp}
        self.rates = self.rates.assign(**new_cols)

    # Computing condition specific predicted counts for all rows of clade counts table
    def predicted_counts_by_clade(self, df):
        # Look-up dictionary condition -> predicted counts
        count_dict = self.rates.set_index("condition")["predicted_count"].to_dict()

        # List of row-wise conditions
        # cond_list = np.column_stack(
        #     [df["mut_type"], df["motif"], df["unpaired"], df["nt_site_before_boundary"]]
        # )
        cond_list = np.column_stack([df["mut_type"], df["motif"]])

        # Vector of predicted counts
        pred_count = np.array(list(map(lambda x: count_dict[tuple(x)], cond_list)))

        return pred_count

    def genome_composition(self, count_df_syn):
        gen_comp = np.array(
            self.rates.apply(
                # lambda x: sum(
                #     (x.mut_type == count_df_syn.mut_type)
                #     & (x.motif == count_df_syn.motif)
                #     & (x.unpaired == count_df_syn.unpaired)
                #     & (
                #         x.nt_site_before_boundary
                #         == count_df_syn.nt_site_before_boundary
                #     )
                # ),
                lambda x: sum(
                    (x.mut_type == count_df_syn.mut_type)
                    & (x.motif == count_df_syn.motif)
                ),
                axis=1,
            )
        )

        return gen_comp

    def evol_time(self, count_df_syn):
        gen_comp = self.genome_composition(count_df_syn)

        N_mut = sum(count_df_syn.actual_count)

        T = N_mut / (gen_comp @ np.array(self.rates.rate))

        return T

    def residual_variance(self, count_df, tau):
        self.rates["residual"] = np.zeros(self.rates.shape[0], float)

        idx = self.rates[self.rates.cond_count != 0].index

        emp_counts = (
            count_df.groupby(
                # ["mut_type", "motif", "unpaired", "nt_site_before_boundary"]
                ["mut_type", "motif"]
            )
            .apply(lambda x: x.actual_count.to_numpy())
            .to_list()
        )

        assert len(idx) == len(emp_counts)

        self.rates.loc[idx, "residual"] = list(
            map(
                lambda x, y: np.mean((np.log(x + 0.5) - np.log(y + 0.5)) ** 2),
                emp_counts,
                self.rates.loc[idx].predicted_count.to_numpy(),
            )
        )

        av_n_cond = np.mean(self.rates.cond_count)

        self.rates["residual"] = (
            self.rates.groupby("mut_type")
            .apply(
                lambda x: np.exp(-x.cond_count / av_n_cond) * tau[x.name]
                + (1 - np.exp(-x.cond_count / av_n_cond)) * x.residual
            )
            .to_list()
        )

    # Populate array with condition specific residuals for each row of clade-wise table of counts
    def residual_by_clade(self, df):
        # Look-up dictionary condition -> residuals
        res_dict = self.rates.set_index("condition")["residual"].to_dict()

        # List of row-wise conditions
        cond_list = np.column_stack(
            # [df["mut_type"], df["motif"], df["unpaired"], df["nt_site_before_boundary"]]
            [df["mut_type"], df["motif"]]
        )

        # Vector of residuals counts
        res = np.array(list(map(lambda x: res_dict[tuple(x)], cond_list)))

        return res


def add_predicted_count(train_df, count_df, clades):
    rate = Rates()

    sites = train_df.nt_site.unique()

    rate.populate_rates(train_df)
    train_df["predicted_count"] = rate.predicted_counts_by_clade(train_df)
    tau = train_df.groupby("mut_type").apply(
        lambda x: np.mean(
            (np.log(x.actual_count + 0.5) - np.log(x.predicted_count + 0.5)) ** 2
        ),
        include_groups=False,
    )
    rate.rates["cond_count"] = rate.genome_composition(train_df)
    rate.residual_variance(train_df, tau)

    for c in clades:
        count_clade = count_df.loc[count_df.clade == c, :].copy()
        count_clade_syn = count_clade.loc[count_clade.mut_class == "synonymous"].copy()
        count_clade_syn = count_clade_syn.loc[
            count_clade_syn.nt_site.isin(sites)
        ]  # retain sites present in training data

        rate.predicted_counts(count_clade_syn)
        pred_count_clade = rate.predicted_counts_by_clade(count_clade)

        residual_clade = rate.residual_by_clade(count_clade)

        count_df.loc[count_clade.index, "predicted_count"] = pred_count_clade
        count_df.loc[count_clade.index, "tau_squared"] = residual_clade


# Add predicted counts to `count_df` for all clades
def add_predicted_count_all_clades(train, count_df):
    clades = count_df.clade.unique()

    add_predicted_count(train, count_df, clades)