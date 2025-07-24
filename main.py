from src.data_cleaning import data_cleaning
from src.data_load import load_data
from src.nlp_wine import nlp_wine_analysis

if __name__ == "__main__":
    df = load_data()
    data_cleaning(df)
