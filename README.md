# LoanRisk — Intelligent Loan Risk Assessment System

An end-to-end machine learning project that takes a loan application from structured inputs to an understandable risk estimate.

LoanRisk brings together data preprocessing, model comparison and a Flask web application. Users can enter applicant and loan details, review the information and view an estimated default probability with supporting indicators.

**[Try the live demo](https://inteliigent-loan-risk-management-sy.vercel.app/)**

> The included dataset is synthetic. The application demonstrates the development and deployment workflow; its results support human review and do not approve or reject loans.

## Features

- Guided assessment with separate loan, applicant and review steps.
- Built-in fictional example for trying the application.
- Estimated default probability and Low, Medium or High risk classification.
- Loan-to-income ratio and rule-based review prompts.
- Separate primary and secondary applicant estimates for education loans.
- Dataset overview with grouped default-rate charts.
- Browser-based assessment history with date and risk filters.
- Editable inputs, printable results, and light and dark themes.
- JSON API for prediction requests.

Education applicants are assessed independently using the same loan terms. Their probabilities are displayed separately and are not combined into a joint score.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript, Jinja |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Model persistence | Joblib |
| Visualization | Matplotlib, browser charts |
| Testing | pytest |
| Assessment history | Browser local storage |

## Machine Learning Workflow

1. Load the structured loan dataset.
2. Separate the predictors from the binary target, `loan_status`.
3. Create an 80/20 stratified train/test split.
4. Fit preprocessing and each classifier using the training subset.
5. Compare Accuracy, Precision, Recall, F1-score and ROC-AUC.
6. Save the pipeline with the highest recorded ROC-AUC.
7. Use the saved pipeline to assess new inputs through Flask.

Numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding. Preprocessing is saved with the classifier to keep training and inference consistent.

### Recorded Results

The bundled comparison uses **3,500 synthetic records**, split into **2,800 training records** and **700 test records**, with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Logistic Regression** | 80.29% | 62.93% | 73.74% | 67.91% | **0.8676** |
| Random Forest | 81.43% | 66.19% | 70.20% | 68.14% | 0.8571 |
| XGBoost | 81.43% | 70.99% | 58.08% | 63.89% | 0.8517 |
| Decision Tree | 75.14% | 54.48% | 73.74% | 62.66% | 0.7866 |

**Logistic Regression is the selected model because it achieved the highest ROC-AUC.** Random Forest and XGBoost achieved slightly higher accuracy, while Random Forest had the highest F1-score.

These values come from [`model_metrics.csv`](Intelligent_Loan_Risk_Assessment_Group17/outputs/model_metrics.csv) and describe the synthetic demonstration experiment.

## Getting Started

### Prerequisites

- Python 3 with the project dependencies supported
- pip
- Git

### Clone the Repository

```bash
git clone https://github.com/Darlington-chinokwetu-codes-dev/Inteliigent-Loan-Risk-Management-System.git
cd Inteliigent-Loan-Risk-Management-System/Intelligent_Loan_Risk_Assessment_Group17
```

The application is inside `Intelligent_Loan_Risk_Assessment_Group17`, so run the following commands from that directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

### Linux or macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:5000** in your browser.

A trained pipeline is included, so retraining is not required to try the application. The local launcher uses Flask’s development server.

## Using the Application

1. Select a loan purpose and enter the amount, interest rate and supplied loan grade.
2. Continue to the applicant information.
3. Enter income, employment, credit-history and debt details.
4. For education loans, complete the secondary applicant section.
5. Review the inputs and select **Assess Loan Risk**.
6. Explore the result, dataset overview or saved assessment history.

Use **Load demo example** to explore the workflow with fictional details.

### Supported Loan Purposes

| Purpose | API value | Applicants |
| --- | --- | --- |
| Education / fees | `EDUCATION` | Primary and secondary |
| Business venture | `VENTURE` | Primary |
| Personal | `PERSONAL` | Primary |
| Debt consolidation | `DEBTCONSOLIDATION` | Primary |
| Home improvement | `HOMEIMPROVEMENT` | Primary |
| Other | `OTHER` | Preview only; no risk score |

The interface uses Indian rupees. Income means **annual income**, and existing debt means the **total outstanding balance**.

### Understanding the Output

| Estimated default probability | Risk category |
| --- | --- |
| Below 30% | Low risk |
| 30% to below 60% | Medium risk |
| 60% or above | High risk |

The binary prediction uses a separate **50% threshold**:

- `0`: Non-default
- `1`: Possible default

The loan-to-income ratio is calculated as:

```text
Loan-to-income ratio = Requested loan amount / Annual income
```

It does not measure monthly repayment affordability. Review suggestions are generated by application rules and are separate from the model’s probability estimate.

## Prediction API

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/predict` | Generate a risk estimate |
| `GET` | `/health` | Check application health |

### Example Request

Send the following JSON to `POST /api/predict`:

```json
{
  "person_age": 31,
  "person_income": 750000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 220000,
  "loan_int_rate": 10.5,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8,
  "credit_score": 720,
  "existing_debt": 85000
}
```

The response includes the default probability, predicted class, risk category, loan-to-income ratio, confidence value and review prompts.

Probability is returned as a percentage. The ratio is returned as a fraction. The `confidence` value is derived from `max(p, 1 - p) × 100`; it is not an independently validated reliability score.

Education requests also require the eight applicant fields with a `secondary_` prefix. Validation errors return **400**, while valid custom-purpose requests that cannot be scored return **422**.

## Retraining

From the application directory, use your virtual environment’s Python interpreter:

```bash
python -m src.train
```

Training reads:

```text
data/raw/loan_risk_demo.csv
```

It updates:

```text
models/loan_risk_pipeline.joblib
models/metadata.json
outputs/model_metrics.csv
```

A replacement dataset must match the feature schema in [`src/train.py`](Intelligent_Loan_Risk_Assessment_Group17/src/train.py), including `loan_percent_income`, and provide the binary `loan_status` target.

## Running Tests

With the virtual environment activated:

```bash
python -m pytest -q
```

The existing tests cover input validation, prediction outputs, education-loan requirements, custom-purpose handling and grouped dataset statistics.

## Project Structure

Paths below are relative to the application directory.

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes and request handling |
| `src/train.py` | Preprocessing, training and model comparison |
| `src/predict.py` | Input preparation and inference |
| `src/risk_utils.py` | Risk categories and review prompts |
| `src/loan_history.py` | Dataset aggregation and chart helpers |
| `src/generate_demo_data.py` | Synthetic data generation |
| `data/raw/` | Training dataset |
| `models/` | Saved pipeline and metadata |
| `outputs/` | Recorded evaluation results |
| `templates/` | Jinja page templates |
| `static/` | Styles, JavaScript and frontend assets |
| `tests/` | Automated tests |
| `requirements.txt` | Python dependencies |

## History and Data Handling

Applicant inputs are sent to Flask for assessment. The history feature stores up to **100 assessment summaries** in the current browser, including assessment time, applicant role, loan-to-income ratio, estimated probability and risk category.

Entered income, debt and credit score are not retained as applicant records in that history.

The dataset overview shows grouped outcomes across synthetic records. Because the dataset has no dates, those charts do not represent changes over time.

## Future Improvements

- Evaluate using an approved, representative dataset.
- Separate model selection from final independent testing.
- Assess probability calibration and subgroup performance.
- Add explanations linked to the model’s predictions.
- Introduce model versioning and drift monitoring.
- Expand accessibility and end-to-end testing.
- Add access controls and a defined retention policy for shared deployments.

## Project Team

Developed as a machine learning project at **Aditya University**.

| Member | Roll Number |
| --- | --- |
| Vajrala Venkata Karthikeyan Reddy | 24B11CS467 |
| Ayodyula Ganga Durga Prasad | 25B21CS087 |
| Pichuka Hemanth | 25B21CS016 |
| Darlington Chinokwetu | 24B11CS538 |

## License

Licensed under the [Apache License 2.0](LICENSE).
