# quantBot: Agentic Trading Strategy Generator

**quantBot** is an interactive dashboard application that uses a local large language model (LLM) to generate, backtest, and interpret equity trading strategies across market regimes. Built using Dash and Plotly, it incorporates cluster-based financial data and realistic backtesting to evaluate long-short strategies driven by fundamental and technical signals.

---

## 🚀 Features

- Agent-powered strategy generation using Ollama-compatible local LLMs (e.g., `gemma3`, `mistral`, `deepseek-r1`)
- Smart feature selection across clustered market regimes
- Fully realistic backtesting with transaction costs and ADV limits
- Interactive visualization of strategy vs benchmark returns
- AI-powered natural language interpretation of backtest results
- Clean, responsive dark-mode UI with side panel regeneration controls

---

## ⚙️ Prerequisites

- Python 3.12.6
- Ollama (with supported models installed)
- Required Python packages listed in `requirements.txt`

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
quantbot/
├── app.py                     # Dash interface
├── agent_service.py          # LLM strategy generation logic
├── backtest_service.py       # Realistic backtester
├── agent_interpretation_service.py  # Markdown-format LLM interpreter
├── data/
│   ├── raw/
│   │   └── crsp.csv
│   ├── processed/
│   │   └── merged_financial_data.csv
│   ├── clustered/
│       ├── clustered_financial_data.csv
│       └── kmeans_cluster_stats.csv
├── download.py               # Initial WRDS/Compustat/CRSP download
├── merge.py                  # Merge downloaded data
├── prep.py                   # Feature engineering and winsorization
├── clustering.py             # KMeans clustering
├── requirements.txt
└── README.md
```

---

## 🧪 Setup Instructions

1. Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd quantbot
pip install -r requirements.txt
```

2. Generate the data (in order):

```bash
python download.py
python merge.py
python prep.py
python clustering.py
```

> **Note:** These scripts download, merge, clean, and cluster the financial data into regime-specific snapshots used for strategy generation.

3. Start the application:

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:8050`.

---

## 📊 Sample Use Case

- Select a model like `gemma3`
- Prompt the agent (e.g., "Focus on momentum-based strategies that avoid volatility")
- Click **Regenerate**
- Review backtested returns, Sharpe ratios, and LLM commentary
- Explore refined prompts and strategy comparison in the dashboard

---

## 🧭 Future Improvements

- Consolidate multi-step preprocessing pipeline
- Add support for multiple LLMs and hyperparameter overrides
- Build strategy library export and backtest comparison tools
- Add agent memory or chain-of-thought refinement iterations

---

## 🧠 Credit
Built with 💻 by Bryn Taylor, ChatGPT 4o, and Claude Sonnet. Powered by scikit-learn, Dash, and Ollama.

---

## 📜 License
This project is licensed for academic and non-commercial use only. For inquiries, please contact the author.