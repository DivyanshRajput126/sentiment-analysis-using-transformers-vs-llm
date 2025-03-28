# Sentiment Analysis: Transformers vs. LLM (BERT vs. GPT-2)

This project performs **Sentiment Analysis** using two different NLP models:
- **BERT (bert-base-uncased)** - A transformer model fine-tuned for sentiment classification.
- **GPT-2** - A large language model (LLM) fine-tuned for sentiment analysis.

The project compares their performance based on **accuracy and confidence** and provides a web interface for real-time sentiment analysis.

---

## 🚀 Features
- **Model Comparison**: Evaluates BERT and GPT-2 on the same dataset.
- **Data Preprocessing**: Text cleaning, tokenization, and preparation for both models.
- **Performance Metrics**: Calculates and displays evaluation metrics(accuracy) for both models.
- **Web Interface**: Interactive web app to test models in real-time.
---

## 📂 Installation & Setup
### 🔹 Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- FastAPI, Uvicorn
- React (for frontend)
- Axios, Tailwind CSS

### 🔹 Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-bert-vs-gpt2.git
cd sentiment-analysis-bert-vs-gpt2
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Model Training & Evaluation
### BERT (Transformer)
- **Model:** bert-base-uncased
- **Training Framework:** PyTorch
- **Metrics:** Accuracy 

### GPT-2 (LLM)
- **Model:** GPT-2
- **Training Framework:** Trainer API
- **Metrics:** Accuracy

---

## 🌐 Web Application
The web interface displays results from both models in real-time, comparing their sentiment analysis capabilities.

### 🔧 Run the Backend
```bash
uvicorn app:app --reload
```

### 🖥️ Run the Frontend
```bash
npm install
npm start
```
Access the app at [http://localhost:3000](http://localhost:3000)

---

## 📈 Results & Performance
| Metric       | BERT-Base-Uncased | GPT-2 |
|--------------|-------------------|-------|
| Accuracy     | 95%               | 68%   |
---

## 🤝 Contributing
Feel free to contribute by submitting **issues** or **pull requests**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments
- Hugging Face for the **Transformers** library
- PyTorch for **deep learning support**

---

Happy coding! 🚀

