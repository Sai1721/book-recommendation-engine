import pandas as pd
import requests
import time

def fetch_description(title, author):
    query = f"{title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'items' in data:
            for item in data['items']:
                volume_info = item.get('volumeInfo', {})
                description = volume_info.get('description')
                if description:
                    return description
        return None
    except Exception as e:
        print(f"Error fetching description for '{title}': {e}")
        return None

def enrich_dataset(csv_path="data/books.csv", max_books=500):
    df = pd.read_csv(csv_path)
    
    if 'description' not in df.columns:
        df['description'] = None
    
    count = 0
    for i, row in df.iterrows():
        if pd.isna(row['description']):
            title = row['title']
            author = row['authors']
            desc = fetch_description(title, author)
            if desc:
                df.at[i, 'description'] = desc
                count += 1
                print(f"[{count}] {title} ✅")
            else:
                print(f"[{count}] {title} ❌ No description found")
            
            time.sleep(1)  # Avoid rate limiting

        if count >= max_books:
            break

    df.dropna(subset=['description'], inplace=True)
    df.to_csv("data/books_with_descriptions.csv", index=False)
    print(f"\n✅ Enrichment complete. {count} books updated.")

if __name__ == "__main__":
    enrich_dataset()
