# 📚 Book Recommendation Engine

A **no-code book recommendation web app** built with **Streamlit**, using **semantic search** via SentenceTransformer embeddings and **FAISS** indexing. Users can input a book title or description, apply filters, and receive personalized book recommendations—complete with covers, genres, ratings, and more.

---

## 🚀 Features

- 🔍 **Semantic Search**: Uses [MiniLM (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to find similar books based on meaning, not just keywords.
- 🎨 **Beautiful UI**: Streamlit frontend with interactive sidebar filters.
- 🎲 **Surprise Me**: Get random book recommendations.
- 🎯 **Filters**: Narrow results by author, genre, and (if available) published year.
- 🌐 **Book Covers**: Integrated with Google Books API to fetch cover images.
- ⭐ **Ratings**: Simulated star ratings with progress bars.
- 📦 **Fast Search**: Powered by FAISS for real-time similarity matching.

---

## 🛠️ Tech Stack

| Component      | Technology                        |
|----------------|-----------------------------------|
| Frontend       | Streamlit                         |
| ML Model       | SentenceTransformer (MiniLM)      |
| Indexing       | FAISS (Facebook AI Similarity Search) |
| Data           | `books_metadata.csv`              |
| External API   | Google Books API                  |
| Hosting        | Streamlit Cloud                   |
| Language       | Python                            |

---

## 📂 Project Structure

```
📁 book-recommender/
├── app.py                  # Main Streamlit application
├── books_metadata.csv      # Dataset (title, author, description, genres, etc.)
├── books.index             # FAISS index file
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

The dataset contains metadata for thousands of books including:

- `title`
- `authors`
- `description`
- `genres` (optional)
- `publishedDate` (if available)

> Note: This project assumes you have already preprocessed the dataset and built a FAISS index based on semantic embeddings of book descriptions/titles.

---

## ⚙️ How It Works

1. User enters a query (book title or description).
2. Model converts the query into an embedding vector.
3. FAISS index retrieves the top-k most similar book vectors.
4. Filtered book metadata is used to enhance results (author, genre).
5. Google Books API fetches cover images.
6. Streamlit displays the results in a clean, user-friendly layout.

---

## 📸 Screenshots

| Home Page | Recommendations |
|-----------|-----------------|
| ![Home](https://via.placeholder.com/300x180?text=Home+Page) | ![Recs](https://via.placeholder.com/300x180?text=Recommendations) |

> Replace the above image links with actual screenshots if available.

---

## 🔧 Installation & Run

1. **Clone the repo**

```bash
git clone https://github.com/Sai1721/book-recommendation-engine.git
cd book-recommendation-engine
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

> Make sure `books_metadata.csv` and `books.index` are in the same folder.

---

## 🌍 Live Demo

Deployed on **Streamlit Cloud**:  
👉 [Click here to use the app](https://book-recommendation-engine.streamlit.app/)

---

## 📌 Future Enhancements

* ✅ Personalized recommendations based on user favorites.
* ✅ Generative AI integration for summarizing or explaining results.
* ⏳ Login and user profiles (optional).
* ⏳ Multi-language support.
* ⏳ Book descriptions fetched dynamically (if missing).

---

## ✍️ Author

**Sairaman Mathivelan**  
AI Intern @ Workcohol | B.Tech - Artificial intelligence and data science

🔗 [LinkedIn](https://www.linkedin.com/in/sairaman-mathivelan-3304b626b/) | 💻 [GitHub](https://github.com/Sai1721)

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

* [SentenceTransformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Google Books API](https://developers.google.com/books)
* [Streamlit](https://streamlit.io/)