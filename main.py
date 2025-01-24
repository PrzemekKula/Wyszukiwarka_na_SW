import re
import sqlite3
import pandas as pd
from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from Levenshtein import distance
from nltk.corpus import stopwords
import nltk


app = Flask(__name__)

# Globalne zmienne
movies_data = None       # DataFrame: [ID, Title, Year, Description]
doc_texts = None         # Lista tekstów (tytuł3x + opis)
doc_sets = None          # Lista zbiorów tokenów (na potrzeby Jaccard)
tfidf_matrix = None      # Macierz TF-IDF
vectorizer = None        # Obiekt TfidfVectorizer
doc_lsi = None           # Macierz wektorów w przestrzeni LSI (n_docs x n_components)
lsi_pipeline = None      # Pipeline: TruncatedSVD + Normalizer
db_path = "C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/movies_database.db"

def load_data_and_build_tfidf():
    """
    Jednorazowo wczytuje dane z bazy i tworzy:
     - doc_texts: tekst z wzmocnionym tytułem
     - doc_sets: zbiór tokenów do Jaccard
     - tfidf_matrix, vectorizer (TF-IDF)
     - doc_lsi, lsi_pipeline (LSI)
    """
    global movies_data, doc_texts, doc_sets
    global tfidf_matrix, vectorizer
    global doc_lsi, lsi_pipeline

    conn = sqlite3.connect(db_path)
    query = "SELECT ID, Title, Year, Description FROM movies"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["Title"] = df["Title"].fillna("")
    df["Description"] = df["Description"].fillna("")

    # Pobierz listę stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))


    # Budujemy listę tekstów (wzmacniamy tytuł 3×)
    texts = []
    sets = []
    for _, row in df.iterrows():
        title = row["Title"]
        description = row["Description"]
        weighted_title = (title + " ") * 3
        full_text = (weighted_title + description).strip()

        # Zapisujemy do doc_texts
        texts.append(full_text)

        # Tworzymy zbiór tokenów do Jaccard (ignorując stopwords)
        tokens = full_text.lower().split()
        filtered_tokens = [t for t in tokens if t not in stop_words]
        sets.append(set(filtered_tokens))

    # TF-IDF
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(texts)

    # LSI = truncated SVD + normalizer
    svd = TruncatedSVD(n_components=200, random_state=42)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)
    X_lsi = pipeline.fit_transform(tfidf)

    # Zapisujemy do zmiennych globalnych
    movies_data = df
    doc_texts = texts
    doc_sets = sets
    vectorizer = vect
    tfidf_matrix = tfidf
    lsi_pipeline = pipeline
    doc_lsi = X_lsi

def highlight_terms(text, query):
    """
    Podkreśla (pogrubia) każde wystąpienie słów z `query` w danym `text`.
    """
    terms = query.split()
    for t in terms:
        pattern = re.compile(r"\b(" + re.escape(t) + r")\b", re.IGNORECASE)
        text = pattern.sub(r"<b>\1</b>", text)
    return text

def jaccard_similarity(setA: set, setB: set) -> float:
    """
    Miara Jaccarda = |A ∩ B| / |A ∪ B|.
    """
    intersec = setA.intersection(setB)
    union = setA.union(setB)
    if len(union) == 0:
        return 0.0
    return len(intersec) / len(union)

def correct_query(user_query, vocabulary):
    """
    Poprawia literówki w zapytaniu użytkownika na podstawie listy słów (vocabulary).
    """
    corrected_words = []
    for word in user_query.split():
        closest_word = min(vocabulary, key=lambda v: distance(word, v))
        corrected_words.append(closest_word)
    return " ".join(corrected_words)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    global movies_data, doc_sets, tfidf_matrix, vectorizer
    global doc_lsi, lsi_pipeline

    if request.method == 'POST':
        user_query = request.form.get('query', '').strip()
        year_min = request.form.get('year_min', '').strip()
        year_max = request.form.get('year_max', '').strip()
        measure = request.form.get('measure', 'cosine')

        # Stwórz słownik słów z danych
        vocabulary = set(" ".join(movies_data['Title']).lower().split())
        vocabulary.update(" ".join(movies_data['Description']).lower().split())

        # Autokorekta zapytania
        corrected_query = correct_query(user_query, vocabulary)
        if corrected_query != user_query:
            message = f"Poprawiono zapytanie: {corrected_query}"
            user_query = corrected_query
        else:
            message = None

        # Filtrowanie danych po roku
        df_filtered = movies_data.copy()
        try:
            if year_min:
                df_filtered = df_filtered[df_filtered["Year"] >= int(year_min)]
            if year_max:
                df_filtered = df_filtered[df_filtered["Year"] <= int(year_max)]
        except ValueError:
            return render_template('search.html', show_form=True, message="Nieprawidłowy zakres lat.")

        if df_filtered.empty:
            return render_template('search.html', show_form=True, message="Brak wyników w podanym zakresie lat.")

        # Dalej: logika similarity i renderowanie wyników
        similarities = []
        filtered_indices = df_filtered.index.tolist()
        query_set = set(user_query.lower().split())
        query_vec = vectorizer.transform([user_query])

        if measure == 'jaccard':
            similarities = [jaccard_similarity(query_set, doc_sets[i]) for i in filtered_indices]
        elif measure == 'lsi':
            query_lsi = lsi_pipeline.transform(query_vec)
            sub_doc_lsi = doc_lsi[filtered_indices, :]
            similarities = cosine_similarity(query_lsi, sub_doc_lsi).flatten().tolist()
        else:
            sub_tfidf_matrix = tfidf_matrix[filtered_indices, :]
            similarities = cosine_similarity(query_vec, sub_tfidf_matrix).flatten().tolist()

        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        results = []
        for local_idx in sorted_indices[:10]:
            real_index = filtered_indices[local_idx]
            row = df_filtered.loc[real_index]
            similarity = similarities[local_idx]
            highlighted_title = highlight_terms(row['Title'], user_query)
            results.append(f"{highlighted_title} ({row['Year']}) [similarity={similarity:.2f}]")

        return render_template('search.html', show_form=True, message=message, results=results)

    return render_template('search.html', show_form=True)

if __name__ == "__main__":
    load_data_and_build_tfidf()
    app.run(debug=True)
