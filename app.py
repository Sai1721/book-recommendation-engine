import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

# Load dataset and index
@st.cache_data
def load_data():
    df = pd.read_csv("books_metadata.csv")
    return df

@st.cache_resource
def load_faiss_index():
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

# Genre filter
if 'genres' in df.columns:
    all_genres = sorted(set(g for sublist in df['genres'].dropna().apply(lambda x: x.split(',')) for g in sublist))
    selected_genres = st.sidebar.multiselect("Select genres", all_genres)
else:
    selected_genres = []

# Author filter
authors = sorted(df['authors'].dropna().unique())
selected_authors = st.sidebar.multiselect("Select authors", authors)

# Year filter
if 'publishedDate' in df.columns:
    years = df['publishedDate'].dropna().apply(lambda x: str(x)[:4] if len(str(x)) >= 4 else None).dropna().unique()
    years = sorted(years)
    selected_years = st.sidebar.multiselect("Select publication years", years)
else:
    selected_years = []

# Filter dataset
filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x for g in selected_genres) if pd.notna(x) else False)]
if selected_authors:
    filtered_df = filtered_df[filtered_df['authors'].isin(selected_authors)]
if selected_years:
    filtered_df = filtered_df[filtered_df['publishedDate'].apply(lambda x: str(x)[:4] if pd.notna(x) else None).isin(selected_years)]

# App Title
st.title("📚 Semantic Book Recommendation Engine")

# Favorite book dropdown
favorite_book = st.selectbox("Optionally, pick your favorite book for personalized results:", [""] + list(df['title'].dropna().unique()))

# Text input for query
query = st.text_input("Enter a book title or description to get recommendations:")

# Book cover fetcher
def fetch_cover(title, author):
    try:
        query = f"{title} {author}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        response = requests.get(url).json()
        if 'items' in response:
            return response['items'][0]['volumeInfo'].get('imageLinks', {}).get('thumbnail', '')
    except:
        return ""
    return ""

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

# Simulate Gemini API explanation
def generate_explanation(user_query, book):
    return f"This book is recommended because it shares similar themes or writing style with your interest in '{user_query}' and has strong alignment in genres like {book['genres']}."

# Recommendation function
def recommend_books(query, k=5, fav_title=None):
    if not query and not fav_title:
        return []

    # Get query embedding
    query_embedding = model.encode([query]) if query else np.zeros((1, 384))

    # Add favorite book embedding if selected
    if fav_title:
        fav_row = df[df['title'] == fav_title]
        if not fav_row.empty:
            fav_idx = fav_row.index[0]
            fav_embedding = index.reconstruct(fav_idx).reshape(1, -1)
            query_embedding = (query_embedding + fav_embedding) / 2

    # Search similar books
    D, I = index.search(np.array(query_embedding), k)
    results = []
    for idx in I[0]:
        row = filtered_df.iloc[idx]
        cover_url = fetch_cover(row['title'], row['authors'])
        rating = round(random.uniform(3.0, 5.0), 2)
        explanation = generate_explanation(query or fav_title, row)
        results.append({
            "title": row['title'],
            "authors": row['authors'],
            "description": row['description'][:300] + "...",
            "cover_url": cover_url,
            "info_link": f"https://books.google.com/books?q={row['title'].replace(' ', '+')}",
            "genres": row['genres'] if 'genres' in row else "",
            "rating": rating,
            "explanation": explanation
        })
    return results

# Show recommendations
if query or favorite_book:
    with st.spinner("Finding recommendations..."):
        recs = recommend_books(query, fav_title=favorite_book)
    if recs:
        cols = st.columns(2)
        for i, book in enumerate(recs):
            with cols[i % 2]:
                st.markdown(f"### {book['title']}")
                st.markdown(f"**Author:** {book['authors']}")
                if book['genres']:
                    genres = [g.strip() for g in book['genres'].split(',')]
                    st.markdown(" ".join([f"<span style='background-color:#eee;border-radius:5px;padding:2px 8px;margin-right:4px'>{g}</span>" for g in genres]), unsafe_allow_html=True)
                if book['cover_url']:
                    st.image(book['cover_url'], width=120)
                st.markdown(f"**Rating:** {book['rating']} ⭐")
                st.progress(int((book['rating'] / 5.0) * 100))
                st.markdown(f"{book['description']}")
                st.markdown(f"💬 *{book['explanation']}*")
                st.markdown(f"[More Info]({book['info_link']})")
                st.markdown("---")
    else:
        st.write("No recommendations found.")
else:
    st.write("Enter a book title or pick a favorite book to get started.")
