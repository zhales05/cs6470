import os
import numpy as np
import pandas as pd

DATA_PATH = "data/housing_data.csv"

def test_dataset_exists():
    assert os.path.exists(DATA_PATH), "Dataset file missing"

def test_load_shapes_and_nans():
    df = pd.read_csv(DATA_PATH)
    X = df[["size", "bedrooms", "age"]].values
    y = df["price"].values

    assert X.ndim == 2 and X.shape[1] == 3
    assert y.ndim == 1
    assert len(X) == len(y)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()