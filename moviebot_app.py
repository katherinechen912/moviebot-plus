# moviebot_app.py
import streamlit as st
from final_multimodal_chatbot import (
    handle_user_input, fetch_movie_poster, get_movie_data,
    apply_grayscale, apply_edge_detection, apply_cartoon_effect,
    apply_movie_poster_effect, apply_vintage_film_effect
)
import os
import re
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="MovieVisionBot",
    page_icon="🎬",
    layout="centered"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5757;
        text-align: center;
        margin-bottom: 20px;
    }
    .bot-message {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #E1F5FE;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .movie-table {
        font-size: 14px;
    }
    .effect-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Function to extract movie title from poster request
def extract_movie_title(text):
    # Remove common words used in poster requests
    text = text.lower()
    remove_words = ["poster", "show", "me", "the", "display", "get", "of", "for"]
    for word in remove_words:
        text = text.replace(word, "")

    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Function to check if input is an image effect request
def is_effect_request(text):
    effect_keywords = ["grayscale", "black and white", "b&w", "edge", "edges", "cartoon",
                       "poster effect", "vintage", "old film", "film effect"]

    return any(keyword in text.lower() for keyword in effect_keywords)


# Function to determine which effect to apply
def get_effect_type(text):
    text = text.lower()

    if any(word in text for word in ["grayscale", "black and white", "b&w"]):
        return "grayscale"
    elif any(word in text for word in ["edge", "edges"]):
        return "edge"
    elif "cartoon" in text:
        return "cartoon"
    elif "poster effect" in text:
        return "poster"
    elif any(word in text for word in ["vintage", "old film"]):
        return "vintage"
    else:
        return None


# Header
st.markdown('<h1 class="main-header">🎬 MovieVisionBot</h1>', unsafe_allow_html=True)
st.markdown("### Your Multimedia Movie Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial bot greeting
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm MovieVisionBot. Ask me about movies, directors, genres, or request a movie recommendation!"
    })

# Initialize with no effect applied
if "current_effect" not in st.session_state:
    st.session_state.current_effect = None

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">MovieVisionBot: {message["content"]}</div>', unsafe_allow_html=True)

# Image display section
if "current_poster" in st.session_state and st.session_state.current_poster:
    # Create a section for the image
    st.subheader("Movie Poster")

    # If an effect has been applied, show the processed image
    if "processed_image" in st.session_state and st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Processed Poster", width=300)

        # Add button to restore original
        if st.button("Restore Original"):
            st.session_state.processed_image = None
            st.session_state.current_effect = None
            st.rerun()
    else:
        # Show the original poster
        st.image(st.session_state.current_poster, caption="Movie Poster", width=300)

    # Show effect buttons
    st.markdown("##### Apply Effects")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Grayscale"):
            # Apply grayscale effect
            processed_path = apply_grayscale(st.session_state.current_poster)
            if processed_path:
                st.session_state.processed_image = processed_path
                st.session_state.current_effect = "grayscale"
                st.rerun()

        if st.button("Cartoon Effect"):
            # Apply cartoon effect
            processed_path = apply_cartoon_effect(st.session_state.current_poster)
            if processed_path:
                st.session_state.processed_image = processed_path
                st.session_state.current_effect = "cartoon"
                st.rerun()

    with col2:
        if st.button("Edge Detection"):
            # Apply edge detection
            processed_path = apply_edge_detection(st.session_state.current_poster)
            if processed_path:
                st.session_state.processed_image = processed_path
                st.session_state.current_effect = "edge"
                st.rerun()

        if st.button("Vintage Film"):
            # Apply vintage effect
            processed_path = apply_vintage_film_effect(st.session_state.current_poster)
            if processed_path:
                st.session_state.processed_image = processed_path
                st.session_state.current_effect = "vintage"
                st.rerun()

# User input
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("Ask me something about movies:",
                               placeholder="Try 'Recommend a movie' or 'Who directed Inception?'")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Check if this is a poster request
        if any(word in user_input.lower() for word in ["poster", "show me", "display"]):
            movie_title = extract_movie_title(user_input)
            poster_path, actual_title = fetch_movie_poster(movie_title)

            if poster_path:
                st.session_state.current_poster = poster_path
                # Reset any effects when loading a new poster
                if "processed_image" in st.session_state:
                    st.session_state.processed_image = None
                st.session_state.current_effect = None
                response = f"Here's the poster for '{actual_title}'."
            else:
                response = f"Sorry, I couldn't find a poster for '{movie_title}'."

        # Check if this is an effect request
        elif is_effect_request(user_input) and "current_poster" in st.session_state:
            effect_type = get_effect_type(user_input)

            if effect_type == "grayscale":
                processed_path = apply_grayscale(st.session_state.current_poster)
                response = "Applied grayscale effect to the poster!"
            elif effect_type == "edge":
                processed_path = apply_edge_detection(st.session_state.current_poster)
                response = "Applied edge detection to the poster!"
            elif effect_type == "cartoon":
                processed_path = apply_cartoon_effect(st.session_state.current_poster)
                response = "Applied cartoon effect to the poster!"
            elif effect_type == "vintage":
                processed_path = apply_vintage_film_effect(st.session_state.current_poster)
                response = "Applied vintage film effect to the poster!"
            else:
                processed_path = None
                response = "I'm not sure which effect you'd like to apply. Try specifying grayscale, edge detection, cartoon, or vintage film."

            if processed_path:
                st.session_state.processed_image = processed_path
                st.session_state.current_effect = effect_type

        # Check if asking about available effects
        elif "what effects" in user_input.lower() or "effects can you" in user_input.lower():
            response = "I can apply these effects to movie posters: grayscale, edge detection, cartoon effect, and vintage film effect. First display a poster, then ask me to apply an effect!"

        # Otherwise get normal response from chatbot
        else:
            response = handle_user_input(user_input)

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # If this is a list request, show formatted movie list
        if "imdb" in user_input.lower() and ("list" in user_input.lower() or "top" in user_input.lower()):
            st.session_state.show_movie_list = True

        # Rerun to update UI
        st.rerun()

# Show movie list if requested
if "show_movie_list" in st.session_state and st.session_state.show_movie_list:
    st.markdown("## IMDb Top Movies")

    # Create a dataframe for better display
    import pandas as pd
    from final_multimodal_chatbot import imdb_top_movies

    # Prepare data for table
    data = []
    for i, movie in enumerate(imdb_top_movies[:25]):  # Show top 25 by default
        data.append({
            "Rank": i + 1,
            "Title": movie['title'],
            "Year": movie['year'],
            "Rating": movie['rating'],
            "Director": movie['director']
        })

    df = pd.DataFrame(data)

    # Display as table
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "Rating": st.column_config.ProgressColumn("Rating", format="%.1f", min_value=0, max_value=10)})

    # Add pagination
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show All Movies"):
            st.session_state.show_movie_list = False
            st.session_state.show_all_movies = True
            st.rerun()

# Show all movies if requested
if "show_all_movies" in st.session_state and st.session_state.show_all_movies:
    st.markdown("## Complete IMDb Top 100")

    import pandas as pd
    from final_multimodal_chatbot import imdb_top_movies

    # Prepare data for table
    data = []
    for i, movie in enumerate(imdb_top_movies):
        data.append({
            "Rank": i + 1,
            "Title": movie['title'],
            "Year": movie['year'],
            "Rating": movie['rating'],
            "Director": movie['director']
        })

    df = pd.DataFrame(data)

    # Display as table with sorting
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "Rating": st.column_config.ProgressColumn("Rating", format="%.1f", min_value=0, max_value=10)})

st.markdown("---")
st.caption("MovieVisionBot | Created by Yujun Katherine Chen")