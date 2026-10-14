# AI-Based Student Performance Prediction

A simple machine learning project that predicts a student's final score.

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- Random Forest Regression

## Project Files
- `app.py` - Main application and ML model
- `student_data.csv` - Sample student dataset
- `requirements.txt` - Required Python packages

## How to Run

1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
streamlit run app.py
```

5. Open the local URL shown by Streamlit in your browser.

## Inputs
- Study hours per day
- Attendance percentage
- Previous score
- Assignments completed

The model predicts the final score from 0 to 100.
