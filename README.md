# PhishDrift
### Explainable Phishing Email Classification with Drift Monitoring and Evidence-Triggered Retraining

PhishDrift is a machine learning project that classifies emails as **phishing or legitimate** while monitoring changes in email characteristics that may indicate model unfamiliarity or performance degradation over time.

Instead of treating phishing detection as a static classification problem, PhishDrift combines multi-model classification, sample-level drift scoring, temporal evaluation, explainability, and evidence-triggered retraining recommendations in an interactive web dashboard.

## Overview

Machine learning models trained on historical email data may become less reliable as phishing tactics, vocabulary, formatting, and legitimate communication patterns evolve. Periodic retraining without evidence of drift can also waste resources or introduce instability.

PhishDrift explores an alternative approach: monitor incoming email characteristics against a calibrated reference, identify unfamiliar samples, and use accumulated drift evidence to recommend when retraining should be considered.

## Key Features

- **Multi-model classification:** Evaluates emails using Multinomial Naive Bayes, Support Vector Machines (SVM), and Random Forest.
- **Email preprocessing and feature engineering:** Cleans email content, extracts domain-specific features, and transforms text into numerical representations.
- **FeSAD-inspired drift monitoring:** Implements sample-level familiarity and drift scoring using the project's calibrated reference data and feature representations.
- **Temporal drift evaluation:** Replays evaluation batches spanning 2019–2025 to investigate changes in classifier behavior over time.
- **Evidence-triggered retraining simulation:** Evaluates drift thresholds and retraining policies instead of relying exclusively on fixed retraining schedules.
- **Explainability:** Presents informative terms, feature contributions, and classification-related evidence to help users interpret predictions.
- **Interactive dashboard:** Provides an interface for email classification, drift monitoring, visualizations, and historical evaluation.
- **Local execution:** Runs through a lightweight Python HTTP server without requiring a separate web framework.

## System Architecture

The application is organized into three main components:

1. **Frontend:** A browser-based interface built with HTML, CSS, and JavaScript.
2. **Server:** A Python HTTP server that handles classification requests, serves dashboard resources, and exposes drift-monitoring endpoints.
3. **Inference and monitoring pipeline:** A Python module responsible for email parsing, preprocessing, model inference, drift calculations, and supporting analysis.

### Workflow

1. Submit email text or upload an email file through the interface.
2. Parse and preprocess the email content.
3. Extract features and obtain predictions from the supported classifiers.
4. Calculate familiarity or drift-related scores using the configured reference information.
5. Display predictions, explanatory information, and monitoring results.
6. Examine temporal evaluation results and retraining-policy recommendations.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | scikit-learn |
| Text processing | TF-IDF and feature engineering |
| Classification | Naive Bayes, SVM, Random Forest |
| Backend | Python standard-library HTTP server |
| Frontend | HTML5, CSS3, JavaScript |
| Visualization | Matplotlib |
| Model persistence | Joblib |
| Evaluation data | CSV |

## Project Structure

```text
phish_drift_ui/
├── artifacts/
│   ├── config.json
│   ├── state_NB_baseline.joblib
│   ├── state_SVM_baseline.joblib
│   ├── state_RF_baseline.joblib
│   ├── detector_eval.csv
│   ├── drift_vs_recall_spearman.csv
│   ├── modern_legit_fp.csv
│   ├── per_sample_Df_frozen.csv
│   ├── policy_simulation.csv
│   ├── policy_summary.csv
│   └── walk_frozen_results.csv
├── web/
│   ├── main.js
│   ├── style.css
│   └── logo.svg
├── charts.py
├── export_inbox.py
├── pipeline.py
├── server.py
├── requirements.txt
├── run.bat
├── README.md
└── PROJECT_DOCUMENTATION.md
```

*The structure above describes the intended organization; some files may be absent depending on the version you publish.*

## Getting Started

### Prerequisites

- Python installed and available from your terminal.
- Git, if you want to clone the repository.
- A modern web browser.

### 1. Clone the repository

```bash
git clone https://github.com/FarishaNA/Phishing_drift_detection.git
cd Phishing_drift_detection
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python and pip executables directly instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the application

From the project directory, run:

```powershell
.\run.bat
```

Alternatively, if the launcher is unavailable, inspect its contents and use the corresponding server command. The application entry point is `server.py`.

Open the local URL printed by the server in your browser.

## Drift Monitoring Approach

PhishDrift includes an implementation inspired by the Feature and Sample Drift (FeSAD) framework.

The monitoring workflow uses calibrated reference information to assess how incoming email samples and evaluation batches differ from the baseline. Individual sample scores help identify unfamiliar inputs, while batch-level analysis supports the evaluation of drift-triggered retraining policies.

The project also includes temporal evaluation artifacts covering yearly batches from 2019 through 2025.

**Important distinction:** Distributional drift and concept drift are related but different. A change in feature distributions alone does not prove that the relationship between email features and phishing labels has changed. The evaluation should therefore be interpreted in terms of the implemented drift measures and the evidence available from the experiments.

## Evaluation

The repository contains evaluation artifacts for:

- Temporal classification performance.
- Drift detection behavior.
- Associations between drift scores and classification performance.
- Sample-level drift analysis.
- Retraining-policy simulations.
- False-positive analysis on legitimate email data.

These artifacts support further investigation into whether drift signals can help identify periods when a classifier may require reassessment or retraining. Results depend on the datasets, preprocessing, model configuration, and evaluation methodology used.

## Privacy and Responsible Use

- Use email data that you have permission to process.
- Avoid committing personal emails, credentials, access tokens, or other sensitive information.
- Review uploaded email content and exported datasets before sharing them.
- Treat predictions as decision-support signals, not definitive proof that an email is malicious or safe.
- Evaluate false positives and false negatives before considering any real-world deployment.

## Limitations

- Evaluation results may not generalize to all email providers, organizations, or phishing campaigns.
- Feature-distribution changes do not independently establish concept drift.
- Drift thresholds depend on the reference data and calibration procedure.
- Retraining recommendations require appropriate validation before being used operationally.
- The project is an experimental research implementation, not a replacement for a complete production email-security system.

## Future Improvements

- Evaluate drift detectors across additional datasets and time periods.
- Investigate relationships between drift signals and changes in classification performance.
- Compare alternative drift metrics and calibration strategies.
- Validate retraining policies using additional temporal and deployment-oriented experiments.
- Improve robustness, monitoring, and reproducibility.

## Acknowledgements

The project is inspired by research on concept drift in machine learning-based phishing detection and the FeSAD framework for feature and sample drift analysis.

## License

A license has not yet been specified. Add an appropriate license file before presenting the repository as open source.
