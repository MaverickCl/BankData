from src.load_data import load_raw_bank_data
from src.clean_data import clean_bank_data, save_processed

def main():
    df = load_raw_bank_data()
    print("Raw shape:", df.shape)

    df_clean = clean_bank_data(df)
    print("Clean shape:", df_clean.shape)
    print(df_clean.isna().sum())

    save_processed(df_clean)
    print("Saved to data/processed/bank_clean.csv")

if __name__ == "__main__":
    main()
