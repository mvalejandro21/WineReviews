from src.data_cleaning import data_cleaning
from src.data_load import load_data


if __name__ == "__main__":
    df = load_data()
    data_cleaning(df)
