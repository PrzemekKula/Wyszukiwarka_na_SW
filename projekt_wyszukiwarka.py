import pandas as pd

file_path = 'C:/Users/DELL/Desktop/ProjektSW/16k_Movies.csv'
movies_df = pd.read_csv(file_path)

"""
movies_df.rename(columns={'Release Date': 'Year'}, inplace=True)
movies_df['Year'] = pd.to_datetime(movies_df['Year'], errors='coerce').dt.year
movies_df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
movies_df = movies_df.drop_duplicates(subset=['Title'])
movies_df.reset_index(drop=True, inplace=True)
movies_df['ID'] = movies_df.index + 1
"""
movies_df['Decade'] = (movies_df['Year'] // 10 * 10).astype('Int64').astype(str) + 's'
updated_file_path = 'C:/Users/DELL/Desktop/ProjektSW/16k_Movies.csv'
movies_df.to_csv(updated_file_path, index=False)



