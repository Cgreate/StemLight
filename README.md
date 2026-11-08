# StemLight

Machine Learning Classification Model Explorer for the Wisconsin Diagnostic Breast Cancer dataset.

The Streamlit app starts with results for 114 held-out test rows. Choose from Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Support Vector Machine. It displays accuracy, AUC, precision, recall, F1, MCC, the confusion matrix, classification report, and downloadable predictions.

The bundled example models were trained separately on 455 rows using a stratified 80/20 split with random state 42. Scaling is stored in the Logistic Regression, KNN, and SVM pipelines. Uploaded CSVs are evaluated without retraining. Labels are 0 = malignant and 1 = benign; precision, recall and F1 use benign as the positive class.

## Run locally

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

- Repository: `Cgreate/StemLight`
- Branch: `main`
- Main file: `app.py`
- Python version in Advanced settings: `3.12`
- GitHub file URL: https://github.com/Cgreate/StemLight/blob/main/app.py

Keep `requirements.txt`, `test_data.csv`, and the `model/` directory in the repository. The saved models require the scikit-learn version pinned in `requirements.txt`.

`stremlight.ipynb` remains the online Jupyter version. Run that notebook in JupyterLite; use `app.py` as the Streamlit deployment entry point.
