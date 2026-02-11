import pandas as pd

df = pd.read_parquet("data/raw/ticker=XOM/year=2016/data.parquet")

def flatten_columns(df):
    new_cols = []
    for col in df.columns:
        # If it looks like a tuple string "(something, something)", extract the first part
        if col.startswith("(") and "," in col:
            col_str = col.split(",")[0].replace("(", "").replace("'", "").strip()
        else:
            col_str = col
        new_cols.append(col_str)
    
    df.columns = new_cols
    
    # Move 'Date' to the front
    if 'Date' in df.columns:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Date')))
        df = df[cols]
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    print(df.head())
    return df

df = flatten_columns(df)
#print(df.columns)
