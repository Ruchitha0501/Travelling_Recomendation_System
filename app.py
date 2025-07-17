import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("google_hotel_data_clean_v2.csv")
    return df

df = load_data()

# Combine features into one column
feature_cols = [f'Feature_{i}' for i in range(1, 10)]
df['all_features'] = df[feature_cols].values.tolist()
df['all_features'] = df['all_features'].apply(lambda x: [i for i in x if pd.notna(i)])

# Generate top 10 most common features
flat_features = [item for sublist in df['all_features'] for item in sublist]
feature_freq = Counter(flat_features)
top_10_features = list(dict(sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)).keys())[:10]

# Human-readable mapping
default_mapping = {
    top_10_features[0]: 'Free Breakfast',
    top_10_features[1]: 'Free WiFi',
    top_10_features[2]: 'Parking',
    top_10_features[3]: 'Pool',
    top_10_features[4]: 'Gym',
    top_10_features[5]: 'Restaurant',
    top_10_features[6]: 'Pet Friendly',
    top_10_features[7]: 'Spa',
    top_10_features[8]: 'Bar',
    top_10_features[9]: 'Air Conditioning'
}

selected_features = list(default_mapping.values())

# Title
st.title("🧭 Travel Recommendation System")
st.markdown("Find hotels based on your preferences in the city of your choice.")

# City selection
unique_cities = sorted(df['City'].dropna().unique())
user_city = st.selectbox("Choose a city:", unique_cities)

# Feature preferences
st.subheader("What features do you want in your hotel?")
user_selections = []
for raw, readable in default_mapping.items():
    user_input = st.checkbox(readable, value=False)
    user_selections.append(1 if user_input else 0)

# Recommendation button
if st.button("Get Recommendations"):

    # Filter by city
    df_city = df[df['City'].str.lower() == user_city.lower()]
    if df_city.empty:
        st.warning("No hotels found for the selected city.")
    else:
        # Keep only 2nd occurrence of each hotel
        df_city['row_number'] = df_city.groupby('Hotel_Name').cumcount()
        df_city = df_city[df_city['row_number'] == 1].drop(columns='row_number')

        # Binary encode selected features
        for raw_feat, mapped_name in default_mapping.items():
            df_city[mapped_name] = df_city['all_features'].apply(lambda x: 1 if raw_feat in x else 0)

        # Final data for similarity
        df_city['Hotel_Rating'] = pd.to_numeric(df_city['Hotel_Rating'], errors='coerce')
        df_city['Hotel_Price'] = pd.to_numeric(df_city['Hotel_Price'], errors='coerce')
        df_final = df_city[['Hotel_Name', 'Hotel_Rating', 'Hotel_Price'] + selected_features].dropna()

        # Compute cosine similarity
        similarity_scores = cosine_similarity([user_selections], df_final[selected_features])[0]
        df_final['Similarity Score'] = similarity_scores

        # Sort and show top 5
        df_top = df_final.sort_values(by='Similarity Score', ascending=False).head(5)

        st.subheader("🏨 Top Recommended Hotels")
        st.dataframe(df_top[['Hotel_Name', 'Hotel_Rating', 'Hotel_Price', 'Similarity Score']].reset_index(drop=True))
