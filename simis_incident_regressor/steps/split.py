"""
This module defines the following routines used by the 'split' step of the regression recipe:
- ``create_dataset_filter``: Defines customizable logic for filtering the training, validation,
  and test datasets produced by the data splitting procedure. Note that arbitrary transformations
  should go into the transform step.
"""

from pandas import DataFrame

def create_dataset_filter(dataset: DataFrame) -> DataFrame:
    """
    Mark rows of the split datasets to be additionally filtered. This function will be called on
    the training, validation, and test datasets.
    :param dataset: The {train,validation,test} dataset produced by the data splitting procedure.
    :return: A Series indicating whether each row should be filtered
    """
    dataset = dataset[["effective_from_utc","status","classification","incident_category","severity_category","region","start_time","incident_cleared_time","suburb","num_lanes_blocked","lateral_position"]]
    #select only rows there status os "Completed"
    dataset = dataset[dataset["status"] == "Completed"]
    #select only rows where classification == "Broken-down Vehicle") | (df.classification == "Primary Crash"
    dataset = dataset[(dataset["classification"] == "Broken-down Vehicle") | (dataset["classification"] == "Primary Crash")]
    #drop status column
    dataset = dataset.drop(columns=["status"])
    #fill in any null values with -1
    dataset = dataset.fillna(-1)

    return dataset
