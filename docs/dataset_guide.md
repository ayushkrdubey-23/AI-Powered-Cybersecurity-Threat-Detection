
# Dataset Guide — AI-Powered Cybersecurity Threat Detection

## 1. Selected Dataset

**Dataset name:** UNSW-NB15 Network Intrusion Detection Dataset

**Official source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Purpose:** Train and evaluate machine learning models that distinguish normal network traffic from potentially malicious traffic.

## 2. Why This Dataset Was Selected

- Developed for network intrusion detection research.
- Contains normal and malicious network traffic records.
- Provides pre-arranged training and testing CSV files.
- Includes attack categories and a binary attack label.
- Supports supervised classification and experiments with anomaly detection.
- Suitable for a student project running on a regular computer.

## 3. Files Required

Download the following prepared dataset files from the official dataset source:

| File | Purpose |
|---|---|
| `UNSW_NB15_training-set.csv` | Training and model development |
| `UNSW_NB15_testing-set.csv` | Independent model evaluation |
| `UNSW-NB15_features.csv` | Optional feature descriptions |

Place the downloaded files directly inside the project's `data/` directory.

Expected structure:

```text
data/
├── UNSW_NB15_training-set.csv
├── UNSW_NB15_testing-set.csv
└── UNSW-NB15_features.csv
```

The feature-description file is optional. The two prepared train/test files are required for the initial implementation.

## 4. Dataset Size

According to the official dataset documentation:

- Training records: 175,341
- Testing records: 82,332
- Total prepared train/test records: 257,673

The original dataset also provides larger collections of network traffic records. We will begin with the prepared split to keep development manageable.

## 5. Important Labels

The prepared files include fields such as:

- `label`: Binary label identifying normal traffic or an attack.
- `attack_cat`: Attack category where applicable.
- `id`: Record identifier in the prepared dataset.

The precise columns will be inspected in the dataset exploration stage.

For binary classification, the intended interpretation is:

- `0`: Normal traffic
- `1`: Attack traffic

For multi-class experiments, `attack_cat` can be used as the target, subject to verification of missing and inconsistent labels.

## 6. Data Handling Rules

1. Preserve the original downloaded data.
2. Never modify the original files manually.
3. Do not use testing labels to train the model.
4. Fit preprocessing transformations using training data only.
5. Evaluate the final model on the prepared testing data.
6. Check for missing values, duplicate records, invalid data types, and class imbalance.
7. Avoid including target labels or record identifiers as ordinary predictive features.
8. Do not claim that the model identifies every real-world cyberattack.

## 7. GitHub and Storage Policy

The raw CSV files are excluded from Git through `.gitignore`.

Do not upload large datasets without checking their source terms, redistribution permissions, and repository size limits.

The repository will contain:

- Dataset source and attribution
- Instructions for obtaining the dataset
- Preprocessing and training code
- Evaluation reports
- Visualizations
- Reproducible instructions

The dataset itself may need to be downloaded separately by anyone reproducing the project.

## 8. Dataset Citation

Primary reference:

Moustafa, N., and Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). Military Communications and Information Systems Conference (MilCIS).

Official dataset documentation:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Follow the dataset provider's academic-use and citation requirements. Commercial use requires permission from the authors according to the official documentation.

## 9. Project Scope

The initial implementation focuses on network intrusion detection, binary attack classification, anomaly detection, and security alert generation.

Phishing email classification, malware file detection, and financial fraud detection are distinct problems. They will not be claimed as implemented merely because the network intrusion model detects attacks.
