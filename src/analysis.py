import pandas as pd

def explore_data():
    df = pd.read_csv("data/processed/bank_clean.csv")

    print("\n================ DATASET OVERVIEW ================")
    print(f"Shape: {df.shape[0]} filas, {df.shape[1]} columnas")

    print("\n================ COLUMNAS =========================")
    for col in df.columns:
        print(f"- {col}")

    print("\n================ TIPOS DE DATOS ==================")
    print(df.dtypes)

    print("\n================ PRIMERAS FILAS ==================")
    print(df.head(5))

    print("\n================ VALORES NULOS ===================")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("No hay valores nulos")
    else:
        print(nulls.sort_values(ascending=False))

    print("\n================ DESCRIPCIÓN NUMÉRICA ============")
    print(df.describe())

    print("\n================ CATEGÓRICAS =====================")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

        

if __name__ == "__main__":
    explore_data()
