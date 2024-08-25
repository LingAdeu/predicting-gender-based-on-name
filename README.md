![header](header.png)

# Using Personal Names to Predict Gender: A 3-Character N-Gram Approach

## About
In this project, I aim to investigate whether a conventional machine learning algorithm along with a character n-grams can outperform Long Short-Term Memory (LSTM) within the character level which already achieved an excellent F1 score of 0.93 (Septiandri, ([2017](https://doi.org/10.48550/arXiv.1707.07129))). To this end, I compared different models with 3-character n-gram, especially within word boundary, so that the models can learn spacing between the name parts (e.g., first and last name). The experiment resulted in Support Vector Machine (SVM) with linear kernel function as the final model, achieving F1 score of 0.94, slightly above the LSTM model's performance. Considering the higher performance over char-LSTM, this project concludes that a conventional model can perform equally well with LSTM model, a type of Recurrent Neural Network (RNN), when a suitable data representation.

## Content
    .
    ├── README.md                   <- The top-level README for using this project
    ├── data
    │   └── indonesian-names.csv    <- The dataset for training and testing the model
    ├── model
    │   └── final_model.pkl         <- The final model (SVM with linear kernel function)
    ├── notebook
    │   └── notebook.ipynb          <- The Jupyter notebook to build the model
    ├── requirements.txt            <- The requirements file for reproducing the environment
    └── src
        └── app.py                  <- Streamlit app

## Feedback
If there are any questions or suggestions for improvements, feel free to contact me here:

<a href="https://www.linkedin.com/in/adelia-januarto/" target="_blank">
    <img src="https://raw.githubusercontent.com/maurodesouza/profile-readme-generator/master/src/assets/icons/social/linkedin/default.svg" width="52" height="40" alt="linkedin logo"/>
  </a>
<a href="mailto:januartoadelia@gmail.com" target="_blank">
    <img src="https://raw.githubusercontent.com/maurodesouza/profile-readme-generator/master/src/assets/icons/social/gmail/default.svg"  width="52" height="40" alt="gmail logo"/>
  </a>