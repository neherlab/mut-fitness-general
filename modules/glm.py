import numpy as np


class GeneralLinearModel:
    def __init__(self, included_factors, W=None, tau_squared=None, ridge=0.1):
        self.included_factors = included_factors
        self.W = W
        self.tau_squared = tau_squared
        self.ridge = ridge

    def train(self, df_train):
        # Initialise dictionaries for model parameters and remaining variances
        self.W = {}
        self.tau_squared = {}

        # Check which mutation types are present in the training data
        present_mut_types = (
            df_train["nt_mutation"].apply(lambda x: x[0] + x[-1]).unique()
        )

        # Make a separate fit for every mutation type
        for mut_type in present_mut_types:
            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            df_mut_type = df_train[
                df_train["nt_mutation"].str.match("^" + nt1 + ".*" + nt2 + "$")
            ]

            # Prepare log-counts, dimensions: (# of sites, 1)
            log_counts = np.log(df_mut_type["actual_count"].values + 0.5).reshape(-1, 1)

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)
            # Fit a linear model as log_counts = w @ X using a mean squared loss with a l2-regularization

            regularization_matrix = self.ridge * np.identity(X.shape[1])
            regularization_matrix[0, 0] = 0  # (don't regularize the offset term)
            w = np.linalg.solve(X.T @ X + regularization_matrix, X.T @ log_counts)

            # Store model parameters
            self.W[mut_type] = w

            # Store remaining variance on log counts
            self.tau_squared[mut_type] = np.mean((log_counts - X @ w) ** 2)


    def create_data_matrix(self, mut_counts_df, mut_type):
        factor_cols = {}

        # Get RNA structure (if available)
        if "unpaired" in mut_counts_df.columns:
            factor_cols["rna_structure"] = [mut_counts_df["unpaired"].values]
        else:
            factor_cols["rna_structure"] = None

        # Get left and right context
        left_context = mut_counts_df["motif"].apply(lambda x: x[0]).values
        right_context = mut_counts_df["motif"].apply(lambda x: x[2]).values
        factor_cols["local_context"] = self.one_hot_l_r(left_context, right_context)

        # Offset term
        base = np.full(len(mut_counts_df), 1)

        # Add columns depending on which factors are included
        columns = [base]
        for factor in self.included_factors:
            if factor_cols[factor] is not None:
                columns += factor_cols[factor]

        # Compile data matrix
        X = np.column_stack(columns)

        return X

    def test(self, df_test):
        mean_sq_errs = {}

        # Check which mutation types are present in the training data
        present_mut_types = (
            df_test["nt_mutation"].apply(lambda x: x[0] + x[-1]).unique()
        )
        # Compute mean squared error separately for every mutation type
        for mut_type in present_mut_types:
            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            df_mut_type = df_test[
                df_test["nt_mutation"].str.match("^" + nt1 + ".*" + nt2 + "$")
            ]

            # Prepare log-counts, dimensions: (# of sites, 1)
            log_counts = np.log(df_mut_type["actual_count"].values + 0.5).reshape(-1, 1)

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)

            # Compute the mean squared error of the fitted model on the training data
            preds = (X @ self.W[mut_type]).flatten()
            log_counts = log_counts.flatten()
            mse = np.mean((log_counts - preds) ** 2)
            mean_sq_errs[mut_type] = mse

        return mean_sq_errs

    def add_predicted_counts(self, df):
        # Check which mutation types are present in the data
        present_mut_types = df["nt_mutation"].apply(lambda x: x[0] + x[-1]).unique()

        # Prepare array for predicted counts and remaining variance
        predicted_counts = np.empty(len(df))
        remaining_variance = np.empty(len(df))

        # Compute mean squared error separately for every mutation type
        for mut_type in present_mut_types:
            # Define ancestral and mutated nucleotide
            nt1, nt2 = mut_type[0], mut_type[1]

            # Select the current mutation type
            mask_mut_type = (
                df["nt_mutation"].str.match("^" + nt1 + ".*" + nt2 + "$").values
            )
            df_mut_type = df[mask_mut_type]

            # Create data matrix X, dimensions: (# of sites, # of parameters in model)
            X = self.create_data_matrix(df_mut_type.copy(), mut_type)

            # Compute the mean squared error of the fitted model on the training data
            predicted_counts[mask_mut_type] = (X @ self.W[mut_type]).flatten()
            remaining_variance[mask_mut_type] = self.tau_squared[mut_type]

        df["predicted_count"] = np.exp(predicted_counts) - 0.5
        df["tau_squared"] = remaining_variance

    @staticmethod
    def one_hot_l_r(context_l, context_r):
        sigma = {}
        for nt in ["C", "G", "T"]:
            sigma[nt + "_l"] = (context_l == nt).astype(int)
            sigma[nt + "_r"] = (context_r == nt).astype(int)

        return [
            sigma["C_l"],
            sigma["G_l"],
            sigma["T_l"],
            sigma["C_r"],
            sigma["G_r"],
            sigma["T_r"],
        ]
