import streamlit as st
import numpy as np
import joblib  # Using joblib instead of pickle

# Load pre-trained models
lr_model_path = 'lasso_model.pkl'  # Update with your local path if needed
scaler_path = 'scaler.pkl'  # Update with your local path if needed

# Load the Lasso regression model and scaler
lr_model = joblib.load(lr_model_path)
scaler = joblib.load(scaler_path)

# Sidebar: Welcome message
st.sidebar.title("Welcome! 🌟")
st.sidebar.markdown("""
Hello there! 👋 Welcome to the **TikTok Like Prediction Chatbot**.  
Here’s what you can do:  
1️⃣ Input your TikTok video data in the form below.  
2️⃣ Get a prediction for your video's expected likes.  
3️⃣ Receive personalized tips to improve engagement!  
Let's get started 🚀
""")

# Center: Main app
st.title("🎥 TikTok Like Predictor & Recommendations")

# Collect user inputs for features
st.header("Input Your TikTok Video Data")
video_view_count = st.number_input("Number of Views", min_value=0, step=1, value=1000)
video_share_count = st.number_input("Number of Shares", min_value=0, step=1, value=50)
video_download_count = st.number_input("Number of Downloads", min_value=0, step=1, value=20)
video_comment_count = st.number_input("Number of Comments", min_value=0, step=1, value=10)
video_duration_sec = st.number_input("Video Duration (seconds)", min_value=1, step=1, value=60)

# Derived features
likes_per_view = video_share_count / (video_view_count + 1e-10)
shares_per_view = video_share_count / (video_view_count + 1e-10)
comments_per_view = video_comment_count / (video_view_count + 1e-10)
downloads_per_view = video_download_count / (video_view_count + 1e-10)
views_likes_interaction = video_view_count * video_share_count

# Feature array
features = np.array([[
    video_view_count,
    video_share_count,
    video_download_count,
    video_comment_count,
    video_duration_sec,
    likes_per_view,
    shares_per_view,
    comments_per_view,
    downloads_per_view,
    views_likes_interaction
]])

# Standardize features
features_scaled = scaler.transform(features)

# Predict using the Lasso Regression model
predicted_likes = lr_model.predict(features_scaled)[0]

# Display prediction
st.subheader("Prediction Results")
st.write(f"The predicted number of likes for this video is approximately **{int(predicted_likes)}**. ❤️")

# Recommendations
st.subheader("📈 Recommendations to Optimize Video Engagement")
if likes_per_view < 0.50:
    st.write("👉 **Boost engagement!** Try using trending sounds and hashtags to increase viewership.")
if shares_per_view < 0.10:
    st.write("👉 **Encourage sharing!** Add captions or calls to action that prompt users to share your video.")
if comments_per_view < 0.15:
    st.write("👉 **Spark conversations!** Ask engaging questions in your video's caption to encourage comments.")
if downloads_per_view < 0.05:
    st.write("👉 **Make it sharable!** Use high-quality visuals and relatable content to encourage downloads.")
if video_duration_sec > 120:
    st.write("👉 **Keep it concise!** Shorter videos often perform better and keep viewers engaged.")

st.markdown("Thank you for using the TikTok Like Prediction Chatbot! 🌟")
