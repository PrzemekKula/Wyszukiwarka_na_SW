import pandas as pd
import sqlite3

file_path = 'C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/16k_Movies.csv'
movies_df = pd.read_csv(file_path)

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

# Zapis danych do bazy SQLite
conn = sqlite3.connect('C:/Users/DELL/Desktop/Wyszukiwarka_na_SW/movies_database.db')
movies_df.to_sql('movies', conn, if_exists='replace', index=False)
conn.close()


