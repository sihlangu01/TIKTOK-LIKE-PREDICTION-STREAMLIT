import streamlit as st
import numpy as np
import pickle

# Load pre-trained models
lasso_model_path = 'C:\Users\vero9\Desktop\TIK TOK PREDICTOR STREAMLIT\TIKTOK-LIKE-PREDICTION-STREAMLIT\lasso_model.pkl'  # Update with your local path if needed
scaler_path = 'C:\Users\vero9\Desktop\TIK TOK PREDICTOR STREAMLIT\TIKTOK-LIKE-PREDICTION-STREAMLIT\scaler.pkl'  # Update with your local path if needed

with open(lasso_model_path, 'rb') as f:
    lasso_model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Sidebar: Welcome message
st.sidebar.title("Welcome! 🌟")
st.sidebar.markdown("""
Hello there! 👋 Welcome to the **TikTok Video Likes Predictor**.  
Here’s what you can do:  
1️⃣ Input your video stats in the form below.  
2️⃣ Get a prediction for how many likes your video might receive.  
3️⃣ Receive personalized tips to maximize likes!  
Let's get started 🚀
""")

# Center: Main app
st.title("🎥 TikTok Likes Predictor & Recommendation Engine")

# Collect user inputs for features
st.header("Input Your Video's Stats")
video_view_count = st.number_input("Video View Count", min_value=0, step=1, value=1000)
video_share_count = st.number_input("Video Share Count", min_value=0, step=1, value=50)
video_download_count = st.number_input("Video Download Count", min_value=0, step=1, value=20)
video_comment_count = st.number_input("Video Comment Count", min_value=0, step=1, value=10)
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

# Predict using the Lasso model
predicted_likes = lasso_model.predict(features_scaled)[0]

# Display prediction
st.subheader("Prediction Results")
st.write(f"Your video is predicted to get approximately **{int(predicted_likes)} likes**. 🎉")

# Recommendations
st.subheader("📈 Recommendations to Improve Likes")
if likes_per_view < 0.02:
    st.write("👉 **Boost engagement!** Try using trending hashtags and engaging captions to increase likes.")
if shares_per_view < 0.01:
    st.write("👉 **Encourage sharing!** Ask viewers to share with friends or include share-worthy content.")
if comments_per_view < 0.005:
    st.write("👉 **Spark conversations!** Pose questions or encourage comments to boost interaction.")
if downloads_per_view < 0.003:
    st.write("👉 **Make it save-worthy!** Create high-quality content that people would want to download.")
if video_duration_sec > 120:
    st.write("👉 **Optimize video length!** Shorter videos (under 1 minute) tend to perform better on TikTok.")

st.markdown("Thank you for using the TikTok Likes Predictor! 🎊 Keep creating amazing content!")
