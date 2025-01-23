import re
import sqlite3
import pandas as pd
from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Globalne zmienne
movies_data = None         # DataFrame z filmami (ID, Title, Year, Description)
tfidf_matrix = None        # Macierz TF-IDF
vectorizer = None          # Obiekt TfidfVectorizer

db_path = "C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/movies_database.db"

def load_data_and_build_tfidf():
    """
    Jednorazowo wczytuje dane z bazy SQLite i tworzy macierz TF-IDF z wzmocnionym tytułem (2x).
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT ID, Title, Year, Description FROM movies"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Uzupełniamy puste wartości
    df["Title"] = df["Title"].fillna("")
    df["Description"] = df["Description"].fillna("")

    # Budujemy listę tekstów do wektoryzacji (wzmacniamy tytuł 2×)
    docs = []
    for _, row in df.iterrows():
        title = row["Title"]
        description = row["Description"]
        weighted_title = (title + " ") * 3  # wzmocnienie tytułu 3x
        full_text = weighted_title + description
        docs.append(full_text)

    # Tworzymy TfidfVectorizer ze stop_words='english'
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(docs)

    return df, tfidf, vect

def highlight_terms(text, query):
    """
    Podkreśla (pogrubia) każde wystąpienie słów z `query` w danym `text` za pomocą <b>...</b>.
    Działa bez rozróżniania wielkości liter i wyszukuje całe słowa (\b).
    """
    terms = query.split()
    for t in terms:
        pattern = re.compile(r"\b(" + re.escape(t) + r")\b", re.IGNORECASE)
        text = pattern.sub(r"<b>\1</b>", text)
    return text

@app.route('/')
def home():
    """
    Strona główna: renderujemy szablon templates/home.html
    """
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    global movies_data, tfidf_matrix, vectorizer

    if request.method == 'POST':
        user_query = request.form.get('query', '').strip()
        year_min = request.form.get('year_min', '').strip()
        year_max = request.form.get('year_max', '').strip()

        # Filtrowanie DataFrame po roku
        df_filtered = movies_data.copy()
        try:
            year_min = int(year_min) if year_min else None
            year_max = int(year_max) if year_max else None
        except ValueError:
            return render_template('search.html',
                                   show_form=True,
                                   message="<p>Nieprawidłowy zakres lat (podaj liczby całkowite).</p>")

        if year_min is not None:
            df_filtered = df_filtered[df_filtered["Year"] >= year_min]
        if year_max is not None:
            df_filtered = df_filtered[df_filtered["Year"] <= year_max]

        if df_filtered.empty:
            return render_template('search.html',
                                   show_form=True,
                                   message="<p>Brak filmów w podanym zakresie lat.</p>")

        # Jeśli zapytanie puste, zwracamy top 10 alfabetycznie
        if not user_query:
            # Sortujemy alfabetycznie (np. aby mieć jakąś kolejność)
            df_filtered = df_filtered.sort_values("Title")
            total_found = len(df_filtered)

            # Budujemy listę (BEZ limitu 10)
            results_list = []
            for _, row in df_filtered.iterrows():
                line = f"{row['Title']} ({row['Year']}) - "
                line += f"<a href='{url_for('movie_details', movie_id=row['ID'])}'>Szczegóły</a>"
                results_list.append(line)

            msg = f"<p>Znaleziono filmów: {total_found} (bez limitu, bo nie podano zapytania).</p>"
            return render_template('search.html',
                                   show_form=True,
                                   message=msg,
                                   results=results_list)

        # Wyszukiwanie TF-IDF
        from scipy.sparse import csr_matrix
        filtered_indices = df_filtered.index.tolist()
        sub_tfidf_matrix = csr_matrix(tfidf_matrix[filtered_indices, :])

        query_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, sub_tfidf_matrix).flatten()

        # Sort malejąco
        ranked_indices = similarities.argsort()[::-1]

        # Tylko filmy z similarity >= 0.4
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] >= 0.4]

        total_found = len(ranked_indices)
        if total_found == 0:
            return render_template('search.html',
                                   show_form=True,
                                   message="<p>Brak filmów z similarity >= 0.4.</p>")

        # Ograniczamy do top 10
        top_10_indices = ranked_indices[:10]

        # Budowa listy wyników
        results_list = []
        for idx_in_sub in top_10_indices:
            real_index = filtered_indices[idx_in_sub]
            row = movies_data.loc[real_index]
            sim_score = similarities[idx_in_sub]
            # Pogrubianie tytułu
            title_with_highlight = highlight_terms(row['Title'], user_query)

            line = f"{title_with_highlight} ({row['Year']}) - [similarity={sim_score:.3f}] - "
            line += f"<a href='{url_for('movie_details', movie_id=row['ID'])}'>Szczegóły</a>"
            results_list.append(line)

        msg = f"<p>Znaleziono w sumie {total_found} filmów (similarity >= 0.4). Pokazuję max 10.</p>"
        return render_template('search.html',
                               show_form=True,
                               message=msg,
                               results=results_list)

    # Metoda GET – wyświetlamy formularz (bez wyników)
    return render_template('search.html', show_form=True)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ID, Title, Year, Description FROM movies WHERE ID = ?", (movie_id,))
    movie = cursor.fetchone()
    conn.close()

    if movie:
        movie_dict = {
            'id': movie[0],
            'title': movie[1],
            'year': movie[2],
            'description': movie[3]
        }
        return render_template('movie.html', movie=movie_dict)
    else:
        return render_template('movie.html', movie=None)

if __name__ == "__main__":
    movies_data, tfidf_matrix, vectorizer = load_data_and_build_tfidf()
    app.run(debug=True)
