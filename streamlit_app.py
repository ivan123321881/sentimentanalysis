import joblib
import streamlit as st

# Load models
lr_model = joblib.load('logistic_regression_model.pkl')
#rf_model = joblib.load('random_forest_model.pkl')
#svm_model = joblib.load('svm_model.pkl')

# App title and description
st.title("Sentiment Analysis System")
st.markdown("""
    **Welcome to the Sentiment Analysis System!**  
    You can input a review below, choose your preferred machine learning model, and the system will determine whether the review is *positive* or *negative*.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Section")

# User Input
user_input = st.sidebar.text_area("Enter your review:", height=150, placeholder="Write your review here...")

# Model Selection
model_option = st.sidebar.selectbox("Choose model:", ("Logistic Regression", "Random Forest", "SVM"))

# Display model accuracy
st.sidebar.subheader("Model Accuracy")
st.sidebar.markdown("""
- Logistic Regression: 73%
- Random Forest: 80%
- SVM: 72%
""")

# Predict sentiment
if st.sidebar.button("Predict"):
    if user_input:
        # Predict using the selected model
        if model_option == "Logistic Regression":
            prediction = lr_model.predict([user_input])
        elif model_option == "Random Forest":
            prediction = rf_model.predict([user_input])
        elif model_option == "SVM":
            prediction = svm_model.predict([user_input])

        # Display the prediction result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("The predicted sentiment is: Positive 🎉")
        else:
            st.error("The predicted sentiment is: Negative 😞")

        # Add some space and user instructions
        st.markdown("---")
        st.markdown("""
        **How does it work?**  
        - The system analyzes the text of the review.
        - Based on the model you choose, it predicts the sentiment.
        - Feel free to experiment with different models to see how each one performs!
        """)
    else:
        st.warning("Please enter a review before clicking predict!")
