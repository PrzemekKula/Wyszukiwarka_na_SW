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
from wordcloud import WordCloud
import os
from flask import send_from_directory

app = Flask(__name__)

# Globalne zmienne
movies_data = None       # DataFrame
doc_texts = None         # Lista tekstów (tytuł 3x + opis)
doc_sets = None          # Lista zbiorów tokenów (na potrzeby Jaccard)
tfidf_matrix = None      # Macierz TF-IDF
vectorizer = None        # Obiekt TfidfVectorizer
doc_lsi = None           # Macierz wektorów w przestrzeni LSI (n_docs x n_components)
lsi_pipeline = None      # Pipeline: TruncatedSVD + Normalizer
db_path = "movies_database.db"
genres_list = None

def load_data_and_build_tfidf():

    global movies_data, doc_texts, doc_sets
    global tfidf_matrix, vectorizer
    global doc_lsi, lsi_pipeline
    global genres_list

    conn = sqlite3.connect(db_path)
    query = "SELECT ID, Title, Year, Description, Genres, Rating, [No of Persons Voted], Decade, [Rating Category], Popularity FROM movies"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["Title"] = df["Title"].fillna("")
    df["Description"] = df["Description"].fillna("")
    df["Genres"] = df["Genres"].fillna("")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["No of Persons Voted"] = pd.to_numeric(df["No of Persons Voted"], errors="coerce").fillna(0)
    df["Decade"] = df["Decade"].fillna("")
    df["Rating Category"] = df["Rating Category"].fillna("")  # Jeśli Decade może zawierać wartości null
    df["Popularity"] = df["Popularity"].fillna("")

    all_genres = set()
    for genres in df["Genres"]:
        for genre in genres.split(","):
            all_genres.add(genre.strip().lower())
    genres_list = sorted(all_genres)  # Posortowana lista gatunków

    # Lista stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    # Lista tekstów (wzmacniamy tytuł 3×)
    texts = []
    sets = []
    for _, row in df.iterrows():
        title = row["Title"]
        description = row["Description"]
        weighted_title = (title + " ") * 3
        full_text = (weighted_title + description).strip()

        # Zapis do doc_texts
        texts.append(full_text)

        # Zbiór tokenów do Jaccard (ignorując stopwords)
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

    # Zmienne globalne
    movies_data = df
    doc_texts = texts
    doc_sets = sets
    vectorizer = vect
    tfidf_matrix = tfidf
    lsi_pipeline = pipeline
    doc_lsi = X_lsi

def highlight_terms(text, query):
    """
    Pogrubia każde wystąpienie słów z zapytania w danym tytule.
    """
    terms = query.split()
    for t in terms:
        pattern = re.compile(r"\b(" + re.escape(t) + r")\b", re.IGNORECASE)
        text = pattern.sub(r"<b>\1</b>", text)
    return text

def jaccard_similarity(setA: set, setB: set) -> float:

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

        # Słownik słów z danych
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
            
            # Filtrowanie po ocenach
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

        # Logika dopasowania wyników
        similarities = []
        filtered_indices = df_filtered.index.tolist()

        # Jeśli podano zapytanie tekstowe
        if user_query:
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
            for local_idx in sorted_indices[:8]:  # Ogranicz do 8 wyników
                real_index = filtered_indices[local_idx]
                row = df_filtered.loc[real_index]
                similarity = similarities[local_idx]
                highlighted_title = highlight_terms(row['Title'], user_query)
                results.append(f"""
                    {highlighted_title} ({row['Year']})
                    <a href="{url_for('movie_details', movie_id=row['ID'])}">
                        <button>Details</button>
                    </a>
                """)
        else:
            # Jeśli nie podano zapytania tekstowego, wyświetl wszystkie pasujące filmy
            results = [
                f"""
                {row['Title']} ({row['Year']})
                <a href="{url_for('movie_details', movie_id=row['ID'])}">
                    <button>Details</button>
                </a>
                """
                for _, row in df_filtered.iterrows()
            ]

        return render_template('search.html', show_form=True, message=message, results=results, genres=genres_list)

    return render_template('search.html', show_form=True, genres=genres_list)

@app.route("/KPIs", methods=["GET", "POST"])
def KPIs_view():
    """
    Endpoint wyświetlający statystyki filmów z możliwością filtrowania po gatunku i
    listę filmów o minimalnym/maksymalnym ratingu.
    """
    global movies_data, genres_list

    if movies_data is None or movies_data.empty:
        return "No movie data available."

    selected_genre = request.form.get("genre", "")
    show_movies_for = request.form.get("show_movies_for", "")  # MinRating lub MaxRating

    # Filtrowanie danych po wybranym gatunku
    if selected_genre:
        filtered_data = (
            movies_data
            .dropna(subset=["Genres"])  # Pomija filmy bez gatunków
            .assign(Genres=lambda df: df["Genres"].str.split(","))  # Rozdziela wielokrotne gatunki
            .explode("Genres")  # Rozdziela wiersze na podstawie wielu gatunków
            .query("Genres.str.strip().str.lower() == @selected_genre.lower()")  # Filtrowanie po wybranym gatunku
        )
    else:
        filtered_data = movies_data

    # KPI – średnia, minimalna i maksymalna ocena (uwzględniają tylko oceny > 0)
    valid_ratings = filtered_data.query("Rating > 0")["Rating"]
    if not valid_ratings.empty:
        avg_rating = round(valid_ratings.mean(), 2)
        min_rating = round(valid_ratings.min(), 2)
        max_rating = round(valid_ratings.max(), 2)
    else:
        avg_rating = min_rating = max_rating = 0

    # Przygotowanie listy filmów o min/max ratingu, jeśli wybrano odpowiednią akcję
    selected_movies = []
    if show_movies_for == "MinRating" and min_rating > 0:
        selected_movies = filtered_data.query("Rating == @min_rating")[["ID", "Title", "Rating"]].to_dict(orient="records")
    elif show_movies_for == "MaxRating" and max_rating > 0:
        selected_movies = filtered_data.query("Rating == @max_rating")[["ID", "Title", "Rating"]].to_dict(orient="records")

    return render_template(
        "KPIs.html",
        total_movies=len(filtered_data),
        avg_rating=avg_rating,
        min_rating=min_rating,
        max_rating=max_rating,
        genres=genres_list,
        selected_genre=selected_genre,
        selected_movies=selected_movies
    )

@app.route('/charts')
def charts_view():
    global movies_data

    # Filtrowanie danych: pomijamy oceny 0
    filtered_data = movies_data[movies_data['Rating'] > 0]

    # Przygotowanie danych dla wykresu kolumnowego
    decade_avg_ratings = (
        filtered_data.groupby('Decade')['Rating'].mean()
        .round(2)  # Zaokrąglamy średnie oceny do dwóch miejsc po przecinku
    )

    # Przygotowanie danych dla wykresu kołowego
    excellent_movies = movies_data[movies_data['Rating Category'] == 'Excellent']

    if not excellent_movies.empty:
        # Rozdzielenie gatunków i zliczenie liczby wystąpień każdego gatunku
        genres_count = (
            excellent_movies['Genres'].str.split(',')
            .explode()  # Rozdziela wielokrotne gatunki na osobne wiersze
            .str.strip()  # Usuwa zbędne spacje
            .value_counts()
        )

        # Obliczenie procentowego udziału
        total_excellent = genres_count.sum()
        genre_percentages = ((genres_count / total_excellent) * 100).round(2)

        # Wybranie 5 największych kategorii
        top_genres = genre_percentages.head(5)
        other_genres_percentage = genre_percentages[5:].sum()

        # Dodanie pozostałych kategorii "Others"
        if other_genres_percentage > 0:
            top_genres["Others"] = other_genres_percentage

        # Przygotowanie danych do wykresu kołowego
        genres = top_genres.index.tolist()
        genre_percentages = top_genres.tolist()
    else:
        genres = []
        genre_percentages = []

    # Przygotowanie danych dla tabeli "Very popular"
    very_popular_movies = movies_data[movies_data['Popularity'] == 'Very popular']
    if not very_popular_movies.empty:
        genre_popularity_counts = (
            very_popular_movies['Genres'].str.split(',')
            .explode()
            .str.strip()
            .value_counts()
        )
        very_popular_data = [
            (genre, count) for genre, count in genre_popularity_counts.items()
        ]
    else:
        very_popular_data = []

    if decade_avg_ratings.empty or not genres:
        return render_template("charts.html", message="No data available for the charts.")

    # Przygotowanie danych do szablonu
    decades = [f"{decade}" for decade in decade_avg_ratings.index]
    avg_ratings = decade_avg_ratings.tolist()

    return render_template(
        "charts.html",
        decades=decades,
        avg_ratings=avg_ratings,
        genres=genres,
        genre_percentages=genre_percentages,
        very_popular_data=very_popular_data  # Dane do tabeli
    )

# Katalog na pliki tymczasowe
TEMP_IMAGE_DIR = "static/temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

@app.route('/cloud', methods=['GET', 'POST'])
def cloud_view():
    global movies_data, genres_list

    selected_genre = request.form.get("genre", "")
    
    # Filtrowanie opisów na podstawie wybranego gatunku
    if selected_genre:
        filtered_data = (
            movies_data[movies_data['Genres'].str.contains(selected_genre, case=False, na=False)]
            .dropna(subset=['Description'])
        )
    else:
        filtered_data = movies_data.dropna(subset=['Description'])

    if filtered_data.empty:
        return render_template("cloud.html", genres=genres_list, message="No descriptions available for this genre.")

    # Generowanie tekstu z kolumny Description
    all_descriptions = ' '.join(filtered_data['Description'])

    # Generowanie chmury słów
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='Set2',
        stopwords=set(stopwords.words('english'))
    ).generate(all_descriptions)

    output_path = "static/wordcloud.png"
    wordcloud.to_file(output_path)

    return render_template(
        "cloud.html",
        genres=genres_list,
        selected_genre=selected_genre,
        wordcloud_img_path=url_for('static', filename='wordcloud.png')
    )

@app.route('/static/temp_images/<filename>')
def serve_temp_image(filename):
    """
    Serwowanie wygenerowanych obrazów z folderu tymczasowego.
    """
    return send_from_directory(TEMP_IMAGE_DIR, filename)

if __name__ == "__main__":
    load_data_and_build_tfidf()
    app.run(debug=True)