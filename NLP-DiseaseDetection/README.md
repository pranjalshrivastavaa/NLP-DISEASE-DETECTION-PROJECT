# 🧠 Disease Detection using BioBERT

This project fine-tunes the BioBERT model for disease detection from natural language symptom descriptions using a custom medical Q&A dataset.

## 📊 Dataset

The dataset (`medDataset_processed.csv`) contains over 16,000 question-answer pairs from medical literature, categorized into:

- Symptoms
- Susceptibility
- Diagnosis
- Treatment
- Other categories

Each entry includes:
- `qtype`: Type of medical question
- `Question`: Natural language medical question
- `Answer`: Expert medical answer

## 🧪 Model

We use [BioBERT](https://github.com/dmis-lab/biobert) (a domain-specific BERT model pre-trained on large-scale biomedical corpora) and fine-tune it for a classification task using the Hugging Face `transformers` library.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/biobert-disease-detection.git
cd biobert-disease-detection

# Install dependencies
pip install -r requirements.txt
