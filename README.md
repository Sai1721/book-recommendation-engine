# 📚 Book Recommendation Engine

A semantic book recommendation engine built with Streamlit, FAISS, and Sentence Transformers. Search for books by title or description and get intelligent recommendations based on semantic similarity. Filter results by genre, author, and publication year.

---

## Features

- **Semantic Search:** Uses Sentence Transformers for meaningful recommendations.
- **Fast Retrieval:** Efficient similarity search with FAISS.
- **Rich Filters:** Filter by genre, author, and year.
- **Book Covers:** Fetches covers from the Google Books API.
- **Interactive UI:** Built with Streamlit.

---

## Project Structure

```
.
├── app.py                      # Main Streamlit app
├── books_metadata.csv          # Book metadata
├── books.index                 # FAISS index of embeddings
├── preprocess.py               # Data and embedding preparation
├── enrich_with_descriptions.py # Add descriptions to books
├── requirements.txt            # Python dependencies
└── ...
```

---

## Usage

1. Enter a book title or description in the search box.
2. Apply filters (genre, author, year) from the sidebar.
3. View recommendations with covers, descriptions, and links.

---

## Data

- Main data: `books_metadata.csv` (columns: `title`, `authors`, `genres`, `publishedDate`, `description`)
- Embeddings indexed in `books.index` using FAISS.

---

## Scripts

- `preprocess.py`: Prepare and embed book data.
- `enrich_with_descriptions.py`: Add descriptions to books.
- `search_books.py`: Command-line search utility.
- `semantic_recommender.py`: Core recommendation logic.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
- [Google Books API](https://developers.google.com/books/)

---

## License

MIT License

---

## Contributing

Pull requests and issues are welcome!

---

## Author

Sairaman Mathivelan

---

**Happy reading! 📚**
