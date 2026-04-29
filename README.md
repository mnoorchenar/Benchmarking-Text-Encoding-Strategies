---
title: Benchmarking-Text-Encoding-Strategies
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🔬 Benchmarking Text Encoding Strategies</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=Surgical+Duration+Prediction+Benchmark;BERT+%2B+Classical+NLP+%2B+Structured+Baselines;K-Fold+%7C+Hospital+CV+%7C+Temporal+CV" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-3b82f6?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🔬 Benchmarking Text Encoding Strategies** — A surgical duration prediction benchmark comparing structured baselines, classical text features, and BERT-based embeddings across multiple ML models and cross-validation strategies.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Dashboard](#-dashboard)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Outputs & Artifacts](#-outputs--artifacts)
- [Reproducibility](#-reproducibility)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>📝 <b>Text Encoding Strategies</b></td>
    <td>Structured-only baseline, Label Encoding, Count Vectorization, TF-IDF, ClinicalBERT, and Sentence-BERT embeddings</td>
  </tr>
  <tr>
    <td>🤖 <b>Multiple ML Models</b></td>
    <td>Linear Regression, Ridge, Lasso, Random Forest, XGBoost, and MLP (TensorFlow/Keras)</td>
  </tr>
  <tr>
    <td>📊 <b>Flexible CV Strategies</b></td>
    <td>Standard K-Fold, Hospital leave-one-location-out CV, and Temporal expanding-window CV</td>
  </tr>
  <tr>
    <td>📈 <b>Comprehensive Metrics</b></td>
    <td>MAE, SMAPE, R², RMSE, training time, and inference time reported per model and encoding</td>
  </tr>
  <tr>
    <td>🖥️ <b>Interactive Dashboard</b></td>
    <td>Flask + Plotly dashboard for exploring and comparing benchmark results interactively</td>
  </tr>
  <tr>
    <td>📄 <b>Publication-Ready Figures</b></td>
    <td>Jupyter notebook exports sensitivity plots and legend PDFs directly to <code>overleaf/</code></td>
  </tr>
</table>

---

## 🏗️ Architecture

<div align="center">
<svg width="600" height="205" viewBox="0 0 600 205" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="205" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
  <text x="300" y="27" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="#1e293b">Benchmarking Text Encoding Strategies</text>
  <rect x="20" y="42" width="140" height="55" rx="8" fill="#3b82f6"/>
  <text x="90" y="66" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Surgical</text>
  <text x="90" y="84" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#bfdbfe">Case Data</text>
  <rect x="205" y="42" width="130" height="55" rx="8" fill="#4f46e5"/>
  <text x="270" y="66" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Text</text>
  <text x="270" y="84" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#c7d2fe">Encoding</text>
  <rect x="380" y="42" width="150" height="55" rx="8" fill="#0ea5e9"/>
  <text x="455" y="66" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Regression</text>
  <text x="455" y="84" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#bae6fd">Models</text>
  <rect x="380" y="140" width="150" height="55" rx="8" fill="#7c3aed"/>
  <text x="455" y="164" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Flask + Plotly</text>
  <text x="455" y="182" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#ede9fe">Dashboard</text>
  <line x1="160" y1="69" x2="197" y2="69" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="197,65 205,69 197,73" fill="#64748b"/>
  <line x1="335" y1="69" x2="372" y2="69" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="372,65 380,69 372,73" fill="#64748b"/>
  <line x1="455" y1="97" x2="455" y2="132" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="451,132 455,140 459,132" fill="#64748b"/>
</svg>
</div>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/Benchmarking-Text-Encoding-Strategies.git
cd Benchmarking-Text-Encoding-Strategies

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scipy scikit-learn optuna xgboost tensorflow torch transformers sentence-transformers matplotlib flask

# 4. Run the main benchmark pipeline
python pipeline.py
```

You will be prompted to run selected stages (or all stages by default).

### Alternative CV Experiments

```bash
python pipeline_cv.py
```

You will be prompted for stage selection, CV strategy, and models.

> **Note:** `pipeline.py` Stage 01 and the BERT cache (Stage 02) must be completed before running `pipeline_cv.py`.

---

## 📊 Dashboard

```bash
python dashboard.py
```

Then open `http://localhost:5050` in your browser. The dashboard reads from `results/result.db`.

---

## 📋 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📊 Results Explorer | Browse and compare model metrics across encoding strategies | ✅ Live |
| 🏥 Hospital CV | Leave-one-location-out cross-validation results | ✅ Live |
| 📅 Temporal CV | Expanding time-series split results | ✅ Live |
| 🔬 Sensitivity Analysis | Sensitivity plots for publication | ✅ Live |
| 📄 Model Comparison | Side-by-side encoding strategy comparison | ✅ Live |
| 📑 Publication Figures | Export PDF figures via `OverLeaf.ipynb` | 🗓️ Notebook |

---

## 🧠 ML Models

```python
# Models benchmarked in this project
models = {
    "linear_regression": "sklearn LinearRegression",
    "ridge":             "sklearn Ridge",
    "lasso":             "sklearn Lasso",
    "random_forest":     "sklearn RandomForestRegressor",
    "xgboost":           "xgboost XGBRegressor",
    "mlp":               "TensorFlow/Keras Sequential (MLP)"
}
```

---

## 📁 Project Structure

```
Benchmarking-Text-Encoding-Strategies/
│
├── 📄 pipeline.py              # Main 4-stage pipeline (K-Fold workflow)
├── 📄 pipeline_cv.py           # Alternative CV pipeline (hospital + temporal)
├── 📄 dashboard.py             # Flask + Plotly interactive dashboard
├── 📄 OverLeaf.ipynb           # Figure generation notebook for publication-ready plots
│
├── 📂 data/
│   ├── 📄 casetime.csv         # Input dataset
│   └── 📂 bert_cache/          # Cached BERT embeddings (.npy)
│
├── 📂 results/                 # Outputs for pipeline.py (result.db, logs, PDFs)
├── 📂 results_hospital/        # Outputs for hospital CV
├── 📂 results_temporal/        # Outputs for temporal CV
├── 📂 overleaf/                # Exported PDF figures/logs from notebook
└── 📄 sync.ps1                 # Optional git sync helper script
```

---

## 📦 Outputs & Artifacts

Main outputs generated by the pipelines:

- **Databases**
  - `data/surgical_data.db` — cleaned data and fold indices
  - `data/fold_encoded.db` — fold-safe encoded feature matrices
  - `results/result.db`, `results_hospital/result.db`, `results_temporal/result.db` — metrics & artifacts
- **Logs**
  - `results/*.log`, `results_hospital/*.log`, `results_temporal/*.log`
- **Figures**
  - Model comparison PDFs in results folders
  - Sensitivity analysis PDFs in `overleaf/`

---

## 🔁 Reproducibility

- Most random operations are configured with fixed seeds in pipeline config blocks.
- Stage 03 operations are fold-aware to reduce leakage risk.
- BERT features are cached so repeated runs are faster and consistent.
- Stages 01–03 auto-skip when already complete; Stage 04 always prompts for model selection.

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available — no real user data is stored. Before publishing this repository publicly, confirm that <code>data/casetime.csv</code> is de-identified and approved for sharing, and verify no sensitive artifacts are present in generated databases or logs. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/Benchmarking-Text-Encoding-Strategies?style=social)](https://github.com/mnoorchenar/Benchmarking-Text-Encoding-Strategies)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/Benchmarking-Text-Encoding-Strategies?style=social)](https://github.com/mnoorchenar/Benchmarking-Text-Encoding-Strategies/fork)

<sub>This project is developed purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
