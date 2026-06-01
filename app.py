import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1.1rem;
    color: #9ca3af;
    text-align: center;
    margin-bottom: 2rem;
}
.movie-card {
    background: rgba(24, 24, 27, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.movie-card:hover {
    transform: translateY(-3px);
    border-color: #ec4899;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.2);
}
.movie-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.movie-year {
    font-size: 0.85rem;
    color: #a1a1aa;
    margin-bottom: 0.8rem;
}
.genre-tag {
    background-color: #1e1b4b;
    border: 1px solid #4338ca;
    color: #c7d2fe;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 50px;
    display: inline-block;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.match-score {
    font-size: 0.8rem;
    font-weight: bold;
    color: #34d399;
    margin-bottom: 0.5rem;
}
div[data-testid="stSlider"] [role="slider"] {
    background-color: #ec4899 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] {
    background: linear-gradient(to right, #ec4899 0%, #ec4899 100%) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
    background-color: #ec4899 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_movie_data():
    df = pd.read_csv("c:/Users/Rakesh/OneDrive/Desktop/ml_project_1/Movie recommendation/movies.csv")
    df = df.dropna(subset=["title", "genres"])
    
    years = []
    clean_titles = []
    genre_lists = []
    genre_sets = []
    
    year_pattern = re.compile(r'\((\d{4})\)')
    
    for title, genre_str in zip(df["title"], df["genres"]):
        match = year_pattern.search(title)
        if match:
            years.append(int(match.group(1)))
            clean_titles.append(year_pattern.sub("", title).strip())
        else:
            years.append(None)
            clean_titles.append(title)
            
        g_list = [g.strip() for g in genre_str.split("|")]
        genre_lists.append(g_list)
        genre_sets.append(set(g_list))
        
    df["year"] = years
    df["clean_title"] = clean_titles
    df["genre_list"] = genre_lists
    df["genre_set"] = genre_sets
    return df

df = load_movie_data()

st.markdown('<div class="main-header">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Content-Based Movie Recommendation Engine</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Recommender System", "📊 Genre Analytics"])

with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    num_recs = st.slider("Max Recommendation Output", 5, 20, 10, step=5)
    st.markdown("---")
    st.markdown("### 🍿 Platform Statistics")
    st.metric("Total Movies in Index", f"{len(df):,}")
    
    all_genres = []
    for g_list in df["genre_list"]:
        all_genres.extend(g_list)
    unique_genres_count = len(set(all_genres))
    st.metric("Tracked Genre Genres", unique_genres_count)

with tab1:
    st.markdown("### 🔮 Content-Based Recommendation Simulator")
    
    search_query = st.text_input("🔍 Type movie keywords to search (e.g. Toy Story, Jumanji, Batman)", value="Toy Story")
    
    if search_query:
        matches = df[df["clean_title"].str.contains(search_query, case=False, na=False)]
        
        if not matches.empty:
            selected_idx = st.selectbox(
                "Select a target baseline movie",
                options=matches.index,
                format_func=lambda idx: f"{df.at[idx, 'clean_title']} ({df.at[idx, 'year'] or 'N/A'})"
            )
            
            target_genres = df.loc[selected_idx, "genre_set"]
            
            st.markdown(f"#### Genres of selected movie:")
            for g in target_genres:
                st.markdown(f'<span class="genre-tag">{g}</span>', unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("### 🏆 Top AI Recommended Movies")
            
            target_genres = df.loc[selected_idx, "genre_set"]
            if target_genres and target_genres != {"(no genres listed)"}:
                matching_df = df[df["genre_set"].apply(lambda s: not s.isdisjoint(target_genres))].copy()
                matching_df = matching_df[matching_df.index != selected_idx]
                
                jaccards = []
                for g_set in matching_df["genre_set"]:
                    union_len = len(target_genres.union(g_set))
                    jaccards.append(len(target_genres.intersection(g_set)) / union_len if union_len > 0 else 0.0)
                    
                matching_df["score"] = jaccards
                recs_df = matching_df.sort_values(by="score", ascending=False).head(num_recs)
                
                if not recs_df.empty:
                    cols = st.columns(2)
                    for i, (idx, row) in enumerate(recs_df.iterrows()):
                        col_idx = i % 2
                        with cols[col_idx]:
                            tags_html = "".join([f'<span class="genre-tag">{g}</span>' for g in row["genre_list"]])
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="match-score">★ {row["score"]*100:.1f}% Match Strength</div>
                                <div class="movie-title">{row["clean_title"]}</div>
                                <div class="movie-year">Year: {row["year"] or "N/A"}</div>
                                <div style="margin-top: 0.5rem;">{tags_html}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No similar movies found inside the index.")
            else:
                st.warning("No genre descriptions available for this selection to compute recommendation vectors.")
        else:
            st.error("No movies matched your keyword search. Try searching for other terms like 'Jurassic', 'Matrix', 'Star Wars'.")

with tab2:
    st.markdown("### 📊 Dataset Genre Frequencies")
    
    counts = pd.Series(all_genres).value_counts().reset_index()
    counts.columns = ["Genre", "Count"]
    
    bar_chart = alt.Chart(counts).mark_bar(color="#ec4899", cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X("Genre:N", sort="-y", title="Genres"),
        y=alt.Y("Count:Q", title="Number of Movies"),
        tooltip=["Genre", "Count"]
    ).properties(height=400)
    
    st.altair_chart(bar_chart, use_container_width=True)


