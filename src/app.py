import streamlit as st
import joblib
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Set page title
st.set_page_config(page_title='Gender Prediction Model')

# Create TextCleaner class
class RemovePunctuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, list):
            return [self.clean_text(text) for text in X]
        elif isinstance(X, pd.Series):
            return X.apply(self.clean_text)
        elif isinstance(X, str):
            return self.clean_text(X)
        else:
            raise ValueError("Unsupported input!")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Load model
path = '../model/final_model.pkl'
model = joblib.load(path)

# Create function to predict
def predict(input_text):
    cleaner = RemovePunctuation()
    cleaned_text = cleaner.transform(input_text)
    data = pd.Series([cleaned_text])
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    return prediction, probability, cleaned_text

# Create function to generate explanation
def lime_explanation(text, seed=42):
    explainer = LimeTextExplainer(class_names=['Female', 'Male'], random_state=seed)
    exp = explainer.explain_instance(text, model.predict_proba, num_features=10)
    return exp

# Build app
def main():
    st.markdown("<h1 style='text-align: left; color: black;'>Name-Based Gender Prediction Model</h1>", 
                unsafe_allow_html=True)
    st.write("This model predicts someone's gender based on their full name")

    text = st.text_area("Enter a **SINGLE** full name here:")

    if st.button("Predict"):
        if not text:
            st.error("Please enter a single full name.")
        else:
            prediction, probability, cleaned_text = predict(text)
            
            prob_male = probability[0][1]
            prob_not_male = probability[0][0]
            
            if prediction[0] == 1:
                st.markdown(f"There is a {prob_male * 100:.2f}% chance the person is <span style='color:#ff7f0f ;font-weight:bold'>MALE</span>.", 
                            unsafe_allow_html=True)
            else:
                st.markdown(f"There is a {prob_not_male * 100:.2f}% chance the person is <span style='color:#1f77b4 ;font-weight:bold'>FEMALE</span>.", 
                            unsafe_allow_html=True)

            # Separate sections
            st.markdown("---")
            st.markdown("#### Explanation on the Prediction")
            st.markdown("""
                The following explanation helps us understand why the model made its prediction. 
                It shows the important name components that influenced the model's decision. 

                - <span style='color:#ff7f0f ;font-weight:bold'>Orange</span> means the name makes it more likely to be <b>MALE</b>.
                - <span style='color:#1f77b4 ;font-weight:bold'>Blue</span> means the name makes it more likely to be <b>FEMALE</b>.
                """, 
                unsafe_allow_html=True)

            # Generate and display explanation as HTML
            exp = lime_explanation(cleaned_text)
            exp_html = exp.as_html()
            components.html(exp_html, height=800)

    st.markdown("<h6 style='text-align: center; color: grey;'>Developed by LingAdeu</h6>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
