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
db_path = "movies_database.db"
genres_list = None

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
    global genres_list  # Dodajemy globalną listę gatunków

    conn = sqlite3.connect(db_path)
    query = "SELECT ID, Title, Year, Description, Genres, Rating, [No of Persons Voted] FROM movies"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["Title"] = df["Title"].fillna("")
    df["Description"] = df["Description"].fillna("")
    df["Genres"] = df["Genres"].fillna("")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["No of Persons Voted"] = pd.to_numeric(df["No of Persons Voted"], errors="coerce").fillna(0)

    # Wyodrębnij unikalne gatunki
    all_genres = set()
    for genres in df["Genres"]:
        for genre in genres.split(","):
            all_genres.add(genre.strip().lower())
    genres_list = sorted(all_genres)  # Posortowana lista gatunków

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
    Podkreśla (pogrubia) każde wystąpienie słów z query w danym text.
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
    Używa dynamicznego progu akceptacji poprawy w zależności od długości słowa.
    """
    corrected_words = []
    for word in user_query.split():
        # Jeśli słowo istnieje w słowniku, nie poprawiaj
        if word in vocabulary:
            corrected_words.append(word)
        else:
            # Znajdź najbliższe słowo w słowniku
            closest_word = min(vocabulary, key=lambda v: distance(word, v))
            
            # Ustal dynamiczny próg akceptacji poprawy
            max_distance = max(2, len(word) // 3)  # Dla krótkich słów tolerancja wynosi co najmniej 2
            if distance(word, closest_word) <= max_distance:
                corrected_words.append(closest_word)
            else:
                # Jeśli nie znaleziono bliskiego dopasowania, zostaw oryginalne słowo
                corrected_words.append(word)
    return " ".join(corrected_words)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    """
    Wyświetla szczegóły filmu na podstawie ID.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Title, Year, Genres, [Duration (minutes)], [Directed by], [Written by], 
            Rating, [No of Persons Voted], Description
        FROM movies
        WHERE ID = ?
    """
    movie = conn.execute(query, (movie_id,)).fetchone()
    conn.close()

    if movie:
        movie_dict = {
            'title': movie[0],
            'year': movie[1],
            'genres': movie[2],
            'duration': movie[3],
            'directed_by': movie[4],
            'written_by': movie[5],
            'rating': movie[6],
            'votes': movie[7],
            'description': movie[8]
        }
        return render_template('movie.html', movie=movie_dict)
    else:
        return render_template('movie.html', movie=None)
    
@app.route('/search', methods=['GET', 'POST'])
def search():
    global movies_data, doc_sets, tfidf_matrix, vectorizer
    global doc_lsi, lsi_pipeline, genres_list

    if request.method == 'POST':
        user_query = request.form.get('query', '').strip()
        year_min = request.form.get('year_min', '').strip()
        year_max = request.form.get('year_max', '').strip()
        genre_filter = request.form.get('genre', '').strip().lower()
        rating_min = request.form.get('rating_min', '').strip()
        rating_max = request.form.get('rating_max', '').strip()
        votes_min = request.form.get('votes_min', '').strip()
        votes_max = request.form.get('votes_max', '').strip()

        measure = request.form.get('measure', 'cosine')
        user_query = user_query.lower()

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

        # Filtracja stopwords w zapytaniu użytkownika
        stop_words = set(stopwords.words('english'))
        user_query_tokens = [t for t in user_query.lower().split() if t not in stop_words]
        user_query = " ".join(user_query_tokens)

        # Filtrowanie danych
        df_filtered = movies_data.copy()
        try:
            # Filtrowanie po roku
            if year_min:
                df_filtered = df_filtered[df_filtered["Year"] >= int(year_min)]
            if year_max:
                df_filtered = df_filtered[df_filtered["Year"] <= int(year_max)]

            # Filtrowanie po gatunkach
            if genre_filter:
                df_filtered = df_filtered[df_filtered["Genres"].str.contains(genre_filter, case=False, na=False)]
            
            if rating_min:
                df_filtered = df_filtered[df_filtered["Rating"] >= float(rating_min)]
            if rating_max:
                df_filtered = df_filtered[df_filtered["Rating"] <= float(rating_max)]

            # Filtrowanie po liczbie głosów
            if votes_min:
                df_filtered = df_filtered[df_filtered["No of Persons Voted"] >= int(votes_min)]
            if votes_max:
                df_filtered = df_filtered[df_filtered["No of Persons Voted"] <= int(votes_max)]
        except ValueError:
            return render_template('search.html', show_form=True, message="Nieprawidłowy wybór filtra.", genres=genres_list)

        if df_filtered.empty:
            return render_template('search.html', show_form=True, message="Brak wyników dla podanych kryteriów.", genres=genres_list)

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
            results.append(f"""
                {highlighted_title} ({row['Year']}) [similarity={similarity:.2f}]
                <a href="{url_for('movie_details', movie_id=row['ID'])}">
                    <button>Details</button>
                </a>
            """)

        return render_template('search.html', show_form=True, message=message, results=results, genres=genres_list)

    return render_template('search.html', show_form=True, genres=genres_list)

@app.route("/statistics")
def statistics_view():
    """
    Endpoint wyświetlający podstawowe statystyki (KPI) oraz wykres (np. rozkład ocen).
    """
    global movies_data

    # Sprawdź, czy dane są wczytane
    if movies_data is None or movies_data.empty:
        return "No movie data loaded."

    # Bierzemy wszystkie dostępne oceny (bez NaN)
    rating_series = movies_data["Rating"].dropna()

    # Tworzymy osobną serię bez zer
    rating_series_no_zero = rating_series[rating_series != 0]

    # Liczba filmów (możesz chcieć wykluczyć też te z oceną = 0, ale to już decyzja biznesowa)
    total_movies = len(movies_data)

    if not rating_series_no_zero.empty:
        # Średnia i minimum tylko dla ocen > 0
        avg_rating = round(rating_series_no_zero.mean(), 2)
        min_rating = rating_series_no_zero.min()
    else:
        # Jeśli nie mamy żadnych ocen > 0, ustawiamy domyślne wartości
        avg_rating = 0
        min_rating = 0

    # Maksymalną ocenę obliczamy z całej serii (uwzględniamy też 0, ale to i tak nie wpłynie na max)
    if not rating_series.empty:
        max_rating = rating_series.max()
    else:
        max_rating = 0

    # Przygotowanie danych do wykresu
    # - jeśli chcesz również pominąć 0 przy wyświetlaniu słupków, użyj rating_series_no_zero
    rating_counts = rating_series_no_zero.apply(lambda x: int(round(x))).value_counts().sort_index()
    chart_labels = rating_counts.index.tolist()  # np. [0, 1, 2, 3, ...]
    chart_values = rating_counts.values.tolist()

    return render_template(
        "statistics.html",
        total_movies=total_movies,
        avg_rating=avg_rating,
        min_rating=min_rating,
        max_rating=max_rating,
        chart_labels=chart_labels,
        chart_values=chart_values
    )


if __name__ == "__main__":
    load_data_and_build_tfidf()
    app.run(debug=True)