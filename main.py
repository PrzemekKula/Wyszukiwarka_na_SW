import pandas as pd
import sqlite3
from flask import Flask, render_template, request

"""
file_path = 'C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/16k_Movies.csv'
movies_df = pd.read_csv(file_path)
"""

def categorize_rating(rating):
    if rating >= 8.0:
        return "Excellent"
    elif rating >= 6.0:
        return "Good"
    else:
        return "Poor"

def categorize_popularity(votes):
    if votes >= 1000:
        return "Very popular"
    elif votes >= 500:
        return "Popular"
    else:
        return " Unpopular"

def convert_duration_to_minutes(duration):
    if pd.isna(duration) or not isinstance(duration, str):
        return None
    try:
        hours, minutes = 0, 0
        if 'h' in duration:
            parts = duration.split('h')
            hours = int(parts[0].strip()) if parts[0].strip() else 0
            minutes = int(parts[1].replace('m', '').strip()) if 'm' in parts[1] and parts[1].strip() else 0
        elif 'm' in duration:
            minutes = int(duration.replace('m', '').strip()) if duration.strip() else 0
        return hours * 60 + minutes
    except (IndexError, ValueError):
        return None

"""
movies_df.rename(columns={'Release Date': 'Year'}, inplace=True)
movies_df['Year'] = pd.to_datetime(movies_df['Year'], errors='coerce').dt.year
movies_df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
movies_df = movies_df.drop_duplicates(subset=['Title'])
movies_df.reset_index(drop=True, inplace=True)
movies_df['ID'] = movies_df.index + 1
movies_df['Decade'] = (movies_df['Year'] // 10 * 10).astype('Int64').astype(str) + 's'
movies_df['Rating Category'] = movies_df['Rating'].apply(categorize_rating)
movies_df['No of Persons Voted'] = movies_df['No of Persons Voted'].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
movies_df['Popularity'] = movies_df['No of Persons Voted'].apply(pd.to_numeric, errors='coerce').apply(categorize_popularity)
movies_df['Duration (minutes)'] = movies_df['Duration'].apply(convert_duration_to_minutes)
movies_df.drop(columns=['Duration'], inplace=True)

updated_file_path = 'C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/16k_Movies.csv'
movies_df.to_csv(updated_file_path, index=False)
"""

"""
# Zapis danych do bazy SQLite
conn = sqlite3.connect('C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/movies_database.db')
movies_df.to_sql('movies', conn, if_exists='replace', index=False)
conn.close()
"""

# Ścieżka do bazy SQLite
db_path = 'C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/movies_database.db'

app = Flask(__name__)

# Strona główna
@app.route('/')
def home():
    return '''
    <h1>Witaj w wyszukiwarce filmów!</h1>
    <p>Wybierz jedną z poniższych opcji:</p>
    <a href="/search"><button>Wyszukaj filmy</button></a>
    <a href="/movie/1"><button>Losowy film</button></a>
    '''

# Endpoint dla wyszukiwania
@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    message = ""
    if request.method == 'POST':
        query = request.form.get('query', '')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE Title LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
        conn.close()

        if not results:
            message = "<p>Nie znaleziono żadnych filmów pasujących do zapytania.</p>"

    return f'''
    <h1>Wyszukiwarka filmów</h1>
    <form method="POST">
        <label for="query">Wpisz tytuł filmu:</label><br>
        <input type="text" id="query" name="query" placeholder="Wpisz tytuł">
        <button type="submit">Szukaj</button>
    </form>
    {message}
    <ul>
        {''.join(f"<li>{row[1]} ({row[2]}) - <a href='/movie/{row[0]}'>Szczegóły</a></li>" for row in results)}
    </ul>
    <a href='/'><button>Powrót na stronę główną</button></a>
    '''

# Endpoint dla szczegółowych informacji o filmie
@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM movies WHERE ID = ?", (movie_id,))
    movie = cursor.fetchone()
    conn.close()

    if movie:
        return f'''
        <h1>Szczegóły filmu</h1>
        <p><strong>Tytuł:</strong> {movie[1]}</p>
        <p><strong>Rok:</strong> {movie[2]}</p>
        <p><strong>Opis:</strong> {movie[3]}</p>
        <a href="/search"><button>Wyszukaj filmy</button></a>
        <a href="/"><button>Powrót na stronę główną</button></a>
        '''
    else:
        return "<h1>Film nie znaleziony</h1><a href='/'><button>Powrót na stronę główną</button></a>"

if __name__ == "__main__":
    app.run(debug=True)
