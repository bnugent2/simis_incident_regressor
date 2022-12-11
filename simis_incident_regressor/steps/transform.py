"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler, FunctionTransformer
from scipy.stats import zscore

def calculate_features(df: DataFrame):
    """
    Extend the input dataframe with pickup day of week and hour, and trip duration.
    Drop the now-unneeded pickup datetime and dropoff datetime columns.
    """
    df['start_time'] = pd.to_datetime(df['start_time'], infer_datetime_format=True)
    df['incident_cleared_time'] = pd.to_datetime(df['incident_cleared_time'], infer_datetime_format=True)
    df['duration'] = round((df['incident_cleared_time'] - df['start_time']) / np.timedelta64(1, 'm'))
    df['start_time'] = df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Australia/Brisbane')
    df['start_time_dayofweek'] = df['start_time'].dt.dayofweek
    df['start_time_hour'] = df['start_time'].dt.hour

    #remlove outliers and 0 duration
    df = df[(np.abs(zscore(df['duration'])) < 2)]
    df = df[df['duration'] != 0]
    return df


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    import sklearn

    function_transformer_params = (
        {}
        if sklearn.__version__.startswith("1.0")
        else {"feature_names_out": "one-to-one"}
    )

    return Pipeline(
        steps=[
            (
                "calculate_time_and_duration_features",
                FunctionTransformer(calculate_features, **function_transformer_params),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "hour_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["region","lateral_position"],
                        ),
                        (
                            "label_encoder",
                            LabelEncoder(),
                            ["classification","incident_category","severity_category","suburb"],
                        ),
                        (
                            "std_scaler",
                            StandardScaler(),
                            ["duration"],
                        ),
                    ]
                ),
            ),
        ]
    )
