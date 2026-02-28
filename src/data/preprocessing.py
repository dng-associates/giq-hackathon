import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def melt_maturities(df: pd.DataFrame, id_vars: tuple = ("data",)):
    """
    Melt the maturities in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with maturities as columns.
        id_vars (tuple): The column(s) to use as identifier variables.
        
    Returns:
        pd.DataFrame: A melted DataFrame with 'Maturity' and 'Value' columns.
    """
    value_vars = [col for col in df.columns if col not in id_vars]
    
    df_long = df.melt(id_vars=id_vars, 
                    value_vars=value_vars,
                    var_name='Maturity',
                    value_name='Value'
    )
    
    df_long["maturity"] = (
        df_long["Maturity"]
        .str.extract(r'(\d+)')
        .astype(float) / 12
    )
    
    return df_long


    def normalize_prices(df: pd.DataFrame, scaler=None):
        """
        Normalize the price values in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with price values.
            value_col (str): The name of the column containing price values to normalize.
        
        Returns:
            pd.DataFrame: A DataFrame with normalized price values.
        """
        if scaler is None:
            scaler = StandardScaler()
            df["price_norm"] = scaler.fit_transform(df[["price"]])
        else:
            df["price_norm"] = scaler.transform(df[["price"]])
        
        return df, scaler
    
    def prepare_features(df: pd.DataFrame):
        """
        Prepare features for modeling.

        Args:
            df (pd.DataFrame): The input DataFrame with raw features.
        Returns:
            pd.DataFrame: A DataFrame with prepared features for modeling.
            Define x and y for modeling.
        """
        X = df[["maturity"]].values
        y = df["price_norm"].values
        
        return X, y