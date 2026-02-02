# Student Depression Prediction App

This project predicts the likelihood of depression in students based on various factors using a Random Forest Classifier.

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    Run the training script to generate the model artifacts (`depression_model.pkl`).
    ```bash
    python train_model.py
    ```

3.  **Run the Application:**
    Start the Streamlit app.
    ```bash
    streamlit run app.py
    ```

## Project Structure
- `app.py`: The main Streamlit application.
- `train_model.py`: Script to train and save the model.
- `requirements.txt`: Python package dependencies.
- `Student Depression Dataset.csv`: Dataset used for training.
