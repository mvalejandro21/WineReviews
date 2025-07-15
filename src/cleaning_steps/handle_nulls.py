import pandas as pd


def handle_nulls(df):
    print(df.isnull().sum())