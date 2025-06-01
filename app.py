import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Custom CSS for attractive background and sidebar ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        min-height: 100vh;
    }
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
        color: #fff !important;
    }
    /* Sidebar text */
    section[data-testid="stSidebar"] .css-1v0mbdj, 
    section[data-testid="stSidebar"] .css-1cpxqw2 {
        color: #fff !important;
    }
    /* Header */
    .st-emotion-cache-10trblm {
        color: #3730a3 !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px #e0e7ff;
    }
    /* Book card */
    .book-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 4px 24px 0 rgba(99,102,241,0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: box-shadow 0.2s;
    }
    .book-card:hover {
        box-shadow: 0 8px 32px 0 rgba(99,102,241,0.18);
    }
    /* Genre badge */
    .genre-badge {
        background: #6366f1;
        color: #fff;
        border-radius: 12px;
        padding: 2px 12px;
        margin-right: 6px;
        font-size: 0.85em;
        display: inline-block;
    }
    /* Progress bar override */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #6366f1, #818cf8);
    }
    </style>
""", unsafe_allow_html=True)
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
st.sidebar.header("🔎 Filter books")

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

st.markdown("<h1 style='text-align:center;'>📚 Semantic Book Recommendation Engine</h1>", unsafe_allow_html=True)

query = st.text_input("🔍 Enter a book title or description to get recommendations:")

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

# Random Book Button
if st.sidebar.button("🎲 Surprise Me!"):
    random_idx = random.choice(filtered_df.index)
    random_book = filtered_df.loc[random_idx]
    st.subheader("🎲 Random Book Recommendation")
    st.markdown(
        f"""
        <div class="book-card">
            <h3>{random_book['title']}</h3>
            <b>Author:</b> {random_book['authors']}<br>
            {" ".join([f"<span class='genre-badge'>{g.strip()}</span>" for g in random_book['genres'].split(',')]) if 'genres' in random_book and pd.notna(random_book['genres']) else ""}
            <p>{random_book['description'][:300]}...</p>
        </div>
        """, unsafe_allow_html=True
    )
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
                st.markdown(
                    f"""
                    <div class="book-card">
                        <h3>{book['title']}</h3>
                        <b>Author:</b> {book['authors']}<br>
                        {" ".join([f"<span class='genre-badge'>{g.strip()}</span>" for g in book['genres'].split(',')]) if book['genres'] else ""}
                        {'<img src="'+book['cover_url']+'" width="120">' if book['cover_url'] else ''}
                        <div style="margin: 8px 0 4px 0;"><b>Rating:</b> {book['rating']} ⭐</div>
                        <div>{book['description']}</div>
                        <a href="{book['info_link']}" target="_blank">More Info</a>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.progress(int((book['rating'] / 5.0) * 100))
    else:
        st.write("No recommendations found for your query.")
else:
    st.write("Enter a book title or description above to get started.")