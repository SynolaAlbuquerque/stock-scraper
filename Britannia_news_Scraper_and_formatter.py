import requests
import pandas as pd


API_KEY = 'lJoJrAEMcfYXMlKKK4q8R4uTWNKkRGypBJOIov1X'  # <- Replace with your Marketaux key
endpoint = 'https://api.marketaux.com/v1/news/all'


search_terms = [
        ('FMCG',25),
('Britannia',25),
        ('Indian market',30),
        ('Britannia quarterly results', 20)

]
language = 'en'
page_size = 100          # Max per request
published_after = '2021-01-03'  # Filter articles after this date


all_articles = []
seen_articles = set()  # Cache to track unique title + date combos


for term, total_pages in search_terms:
   for page in range(1, total_pages + 1):
       params = {
           'api_token': API_KEY,
           'search': term,
           'language': language,
           'limit': page_size,
           'page': page,
           'published_after': published_after
       }
       response = requests.get(endpoint, params=params)
       if response.status_code == 429:
           print("Rate/credit limit reached. Try again later or upgrade your plan.")
           break
       elif response.status_code != 200:
           print('Error:', response.status_code, response.text)
           continue
       articles = response.json().get('data', [])
       for article in articles:
           title = article.get('title', '')
           published_at = article.get('published_at', '')
           unique_key = (title.lower().strip(), published_at)  # Case-insensitive title + exact date
           if unique_key not in seen_articles:
               seen_articles.add(unique_key)
               all_articles.append({
                   'title': title,
                   'description': article.get('description'),
                   'published_at': published_at,
                   'url': article.get('url'),
                   'source': article.get('source')
               })


df = pd.DataFrame(all_articles)


# Final dedup (redundant but ensures cleanliness)
df.drop_duplicates(subset=['title', 'published_at'], inplace=True)


df.to_csv('Britannia_news_scraped.csv', index=False)


print(f"Fetched {len(df)} unique articles after deduplication.")
df.head()
df.Download('Britannia_news_scraped.txt')

'''
=============================================================================================================================================
                       Step 2 Parsing headlines
=============================================================================================================================================
'''
import re

def parse_headlines(df):
    records = []
    for index, row in df.iterrows():
        title = row.get('title', '')
        published_at_str = row.get('published_at', '')
        source = row.get('source', '')

        date = None
        if published_at_str:
            try:
                # Assuming the published_at is in the format 'YYYY-MM-DDTHH:MM:SS.SSSSSSZ'
                date = pd.to_datetime(published_at_str, errors='coerce')
            except Exception as e:
                print(f"Error parsing date '{published_at_str}': {e}")
                date = None

        if title and date is not None and source:
            records.append({'date': date, 'headline': title.strip(), 'source': source.strip()})

    return pd.DataFrame(records)

news_df = parse_headlines(df)
# Ensure news_df has both date and source columns before dropping NA
news_df = news_df.dropna(subset=['date', 'source']).sort_values('date').reset_index(drop=True)
print(f"Parsed {len(news_df)} headlines with dates and sources.")
display(news_df.head())
