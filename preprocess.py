import pandas as pd

def load_and_clean_data(csv_path="data/books.csv"):
    df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1, encoding='utf-8')

    df = df[['title', 'authors', 'average_rating', 'language_code', 'description']]
    df.dropna(subset=['description'], inplace=True)
    df = df[df['description'].str.len() > 100]
    df.drop_duplicates(subset=['title', 'authors'], inplace=True)
    df = df[df['language_code'] == 'eng']
    df.reset_index(drop=True, inplace=True)

    print(f"Cleaned dataset contains {len(df)} books.")
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head(3))
