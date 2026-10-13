
# System Architecture and Reproducibility

## 1. Overview

The AI-Powered Cybersecurity Threat Detection System is a modular Python project for offline network intrusion detection.

It separates dataset ingestion, cleaning, feature engineering, model training, evaluation, detection, reporting, visualization, and dashboard presentation.

## 2. Processing Architecture

```text
UNSW-NB15 CSV Files
        |
        v
src/data_loader.py
        |
        v
src/preprocessing.py
        |
        v
Feature/Target Separation
        |
        v
src/feature_engineering.py
        |
        v
Fitted Preprocessing Transformer
        |
        +-------------------------------+
        |                               |
        v                               v
src/model_training.py           Saved Transformer
        |                               |
        v                               v
Trained Classifiers              Inference Input
        |                               |
        v                               v
src/model_evaluation.py          src/threat_detection.py
        |                               |
        v                               v
Evaluation Reports               Threat Alerts
        |                               |
        +---------------+---------------+
                        |
                        v
              src/visualization.py
                        |
                        v
               outputs/visualizations/
                        |
                        v
                app/dashboard.py
```

## 3. Module Responsibilities

### Dataset loading

`src/data_loader.py` loads the prepared training and testing CSV files and summarizes dataset dimensions, data types, label distributions, missing values, and duplicate rows.

### Preprocessing

`src/preprocessing.py` standardizes common missing-value markers, handles infinite numeric values, validates the binary label, and separates features from target labels.

### Feature engineering

`src/feature_engineering.py` uses a Scikit-learn ColumnTransformer.

Numerical features use median imputation and standardization.

Categorical features use constant-value imputation and one-hot encoding with unknown-category handling.

The transformer must be fitted on training data only. The same fitted transformer is reused for evaluation and inference.

### Model training

`src/model_training.py` trains Logistic Regression and Random Forest classifiers on the transformed training data.

The fitted transformer and trained models are serialized locally in `models/`.

### Model evaluation

`src/model_evaluation.py` evaluates the saved models on the prepared testing dataset without refitting either the transformer or the models.

It writes evaluation metrics, classification reports, confusion matrices, and prediction outputs.

### Threat detection

`src/threat_detection.py` uses the saved Random Forest model to generate attack scores and severity-ranked alert records.

Ground-truth labels are excluded from predictive inputs. They are not used to make the model's predictions.

### Visualization

`src/visualization.py` generates charts from the actual saved evaluation and detection results.

### Dashboard

`app/dashboard.py` displays generated metrics, alert records, and graphs through Streamlit.

The dashboard is a reporting interface rather than a live network sensor.

## 4. Reproducible Execution Order

Run all commands from the repository root with the project virtual environment activated.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run unit tests:

```powershell
python -m pytest -v
```

Train the models:

```powershell
python -m src.model_training
```

Evaluate the models:

```powershell
python -m src.model_evaluation
```

Generate threat alerts:

```powershell
python -m src.threat_detection
```

Generate visualizations:

```powershell
python -m src.visualization
```

Launch the dashboard:

```powershell
python -m streamlit run app/dashboard.py
```

The dataset must be downloaded separately unless a future repository version provides an explicitly authorized data-acquisition mechanism.

## 5. Training and Testing Separation

The prepared training CSV is used to fit the preprocessing transformer and classifiers.

The prepared testing CSV is used only for independent evaluation and offline detection.

Testing labels may be used later to compare predictions against ground truth during evaluation, but they must not be included as model input features or used to refit preprocessing.

All data transformations used during inference must match the fitted training pipeline.

## 6. Output Directory

`outputs/` contains generated evaluation reports, predictions, detection records, alerts, summaries, and charts.

`models/` contains locally generated serialized model artifacts.

These generated artifacts are excluded from Git by default. Reproduce them by executing the documented commands.

## 7. Reproducibility Notes

Random states are set for the classifiers to improve repeatability.

Exact results can still depend on library versions, dataset versions, operating environment, and model configuration.

For a stronger reproducibility record, document the installed Python version, dependency versions, dataset source, model configuration, and actual evaluation metrics when conducting an experiment.

## 8. Limitations

The system is an offline educational simulation.

It does not capture live packets, access corporate infrastructure, execute containment actions, or establish that an alert corresponds to a confirmed incident.

Model scores and heuristic severity levels must not be treated as calibrated real-world compromise probabilities.
