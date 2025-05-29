import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load dataset and index
@st.cache_data
def load_data():
    df = pd.read_csv("books_metadata.csv")
    return df

@st.cache_resource
def load_faiss_index():
    import faiss
    index = faiss.read_index("books.index")
    return index

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

df = load_data()
index = load_faiss_index()
model = load_model()

# Sidebar filters
st.sidebar.header("Filter books")

# Filter: Genre
if 'genres' in df.columns:
    all_genres = sorted(set(g for sublist in df['genres'].dropna().apply(lambda x: x.split(',')) for g in sublist))
    selected_genres = st.sidebar.multiselect("Select genres", all_genres)
else:
    selected_genres = []

# Filter: Author
authors = sorted(df['authors'].dropna().unique())
selected_authors = st.sidebar.multiselect("Select authors", authors)

# Filter: Year
if 'publishedDate' in df.columns:
    years = df['publishedDate'].dropna().apply(lambda x: str(x)[:4] if len(str(x))>=4 else None).dropna().unique()
    years = sorted(years)
    selected_years = st.sidebar.multiselect("Select publication years", years)
else:
    selected_years = []

# Filter data based on selections
filtered_df = df.copy()

if selected_genres:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x for g in selected_genres) if pd.notna(x) else False)]
if selected_authors:
    filtered_df = filtered_df[filtered_df['authors'].isin(selected_authors)]
if selected_years:
    filtered_df = filtered_df[filtered_df['publishedDate'].apply(lambda x: str(x)[:4] if pd.notna(x) else None).isin(selected_years)]

st.title("📚 Semantic Book Recommendation Engine")

query = st.text_input("Enter a book title or description to get recommendations:")

def fetch_cover(title, author):
    try:
        query = f"{title} {author}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        response = requests.get(url).json()
        if 'items' in response:
            thumbnail = response['items'][0]['volumeInfo'].get('imageLinks', {}).get('thumbnail', '')
            return thumbnail
    except:
        return ""
    return ""
import random

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #222; color: #fff; }
        .st-bb { background-color: #333 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Random Book Button
if st.sidebar.button("🎲 Surprise Me!"):
    random_idx = random.choice(filtered_df.index)
    random_book = filtered_df.loc[random_idx]
    st.subheader("🎲 Random Book Recommendation")
    st.markdown(f"### {random_book['title']}")
    st.markdown(f"**Author:** {random_book['authors']}")
    if 'genres' in random_book and pd.notna(random_book['genres']):
        genres = [g.strip() for g in random_book['genres'].split(',')]
        st.markdown(" ".join([f"<span style='background-color:#eee;border-radius:5px;padding:2px 8px;margin-right:4px'>{g}</span>" for g in genres]), unsafe_allow_html=True)
    st.markdown(f"{random_book['description'][:300]}...")
    st.markdown("---")
    
def recommend_books(query, k=5):
    if not query:
        return []
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    results = []
    for idx in I[0]:
        row = filtered_df.iloc[idx]
        cover_url = fetch_cover(row['title'], row['authors'])
        # Simulate a rating for demo purposes
        rating = round(random.uniform(3.0, 5.0), 2)
        results.append({
            "title": row['title'],
            "authors": row['authors'],
            "description": row['description'][:300] + "...",
            "cover_url": cover_url,
            "info_link": f"https://books.google.com/books?q={row['title'].replace(' ', '+')}",
            "genres": row['genres'] if 'genres' in row else "",
            "rating": rating
        })
    return results

if query:
    with st.spinner("Finding recommendations..."):
        recs = recommend_books(query)
    if recs:
        cols = st.columns(2)
        for i, book in enumerate(recs):
            with cols[i % 2]:
                st.markdown(f"### {book['title']}")
                st.markdown(f"**Author:** {book['authors']}")
                # Genre badges
                if book['genres']:
                    genres = [g.strip() for g in book['genres'].split(',')]
                    st.markdown(" ".join([f"<span style='background-color:#eee;border-radius:5px;padding:2px 8px;margin-right:4px'>{g}</span>" for g in genres]), unsafe_allow_html=True)
                # Book cover
                if book['cover_url']:
                    st.image(book['cover_url'], width=120)
                # Rating bar
                st.markdown(f"**Rating:** {book['rating']} ⭐")
                st.progress(int((book['rating'] / 5.0) * 100))
                st.markdown(f"{book['description']}")
                st.markdown(f"[More Info]({book['info_link']})")
                st.markdown("---")
    else:
        st.write("No recommendations found for your query.")
else:
    st.write("Enter a book title or description above to get started.")
