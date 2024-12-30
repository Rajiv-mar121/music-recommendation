import numpy as np
import pandas as pd
import RecommendersModel as recommenders

print("Rajiv")
print(np.__version__)

song_relaion_df = pd.read_csv('data/triplets_file.csv')
print(song_relaion_df.head())

song_data_df = pd.read_csv('data/song_data.csv')
print(song_data_df.head())

# Merge both the data frames
# combine both data
song_df = pd.merge(song_relaion_df, song_data_df.drop_duplicates(['song_id']), on='song_id', how='left')
print(song_df.head())
print(song_df.shape)


print(song_df['title'][1])

## Now create new feature/column as song by adding title and artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name']
print(song_df.head())


print(len(song_df))  ## total records 20,00,000 2 Million 20 Lakhs

# taking top 10k samples for quick results
song_df = song_df.head(10000)

# cummulative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
print(song_grouped.head())

# Now Calculate percentage
total_listen_count= song_grouped['listen_count'].sum()
song_grouped['percentage(%)']  = (song_grouped['listen_count']/total_listen_count)*100

song_grouped = song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
print(song_grouped)

## Popularity Recommendation Engine
print(song_df['user_id'][5])
pr = recommenders.popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')

# display the top 10 popular songs
recommended = pr.recommend(song_df['user_id'][5])
print(recommended)

# display the top 10 trending songs
trending = pr.recommendtrendingsong(song_df, 'listen_count', 'song')
print(trending)

## Item Similarity Recommendation

similar_recommendation = recommenders.item_similarity_recommender_py()
similar_recommendation.create(song_df, 'user_id', 'song')

user_items = similar_recommendation.get_user_items(song_df['user_id'][5])


# display user songs history
for user_item in user_items:
    print(user_item)

similar_recommneded_song = similar_recommendation.recommend(song_df['user_id'][5])
print(similar_recommneded_song)