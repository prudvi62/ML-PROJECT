# Customer Credit Scoring — Frontend

A Streamlit web frontend for the Customer Credit Scoring project. It trains an XGBoost model from the project's `cs-training.csv` dataset and lets users enter customer features to estimate the probability of `SeriousDlqin2yrs = 1`.

## Run locally

1. Install Python 3.10 or newer.
2. Extract this folder and open a terminal inside it.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the web app:

   ```bash
   streamlit run app.py
   ```

5. In the app's left sidebar, upload `cs-training.csv` (the Give Me Some Credit dataset used in the notebook).
6. After training finishes, enter the customer's ten feature values and select **Assess credit risk**.

## Important notes

- The CSV must contain `SeriousDlqin2yrs` and the ten feature columns listed in the notebook.
- Model training runs in the app session and is cached for that session; first load may take a little while.
- Missing feature values are imputed with medians learned from the training split, avoiding fitting the imputer on the test split.
- The positive class means serious delinquency within two years. The 0.50 decision threshold is a demo default and should be tuned/validated for the intended use.
- This is an academic prototype, not a production lending decision tool.
