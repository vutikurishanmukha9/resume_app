# 🧠 Resume Analyzer – AI-Powered Resume Intelligence (resume_app)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered web app that **analyzes resumes, predicts job roles, matches job descriptions, and estimates salaries** using a combination of **machine learning** and **natural language processing (NLP)**.  
Designed for intelligent career insights — fast, interpretable, and lightweight.

---

## 🚀 Overview

The **Resume Analyzer** is built using **Flask** for the backend and **HTML/CSS/JavaScript** for the frontend.  
It performs three main tasks:
1. **Predicts the best-fit job category** from your resume.
2. **Finds semantically similar job descriptions** using sentence embeddings.
3. **Estimates an expected salary range** using a trained regression model.

It’s like your personal career analyst — but instant and open-source.

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Flask (Python) |
| **ML/NLP Models** | scikit-learn, Sentence Transformers |
| **Vectorization** | TF-IDF |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Data Handling** | pandas, numpy |
| **Persistence** | joblib |
| **PDF Parsing** | PyPDF2 |
| **Environment** | Virtualenv (venv) |
| **Version Control** | Git + GitHub |

---

## 🧠 Core Features

- **Smart Resume Analysis:** Reads your resume text or PDF and extracts relevant skills & context.  
- **Job Category Prediction:** Uses a Logistic Regression model trained on real job data.  
- **Semantic Job Matching:** Leverages `SentenceTransformer` to match resumes with top job descriptions.  
- **Salary Prediction:** Uses a trained Linear Regression model to estimate expected pay.  
- **Fast Local Web App:** Simple Flask setup, runs locally with zero latency.  
- **Easily Extendable:** Add your datasets or retrain models for custom job domains.

---

## ⚙️ How It Works

1. **Input Resume** (paste text or upload PDF)  
2. **TF-IDF Vectorizer** encodes your resume  
3. **Job Classifier** predicts the most probable category  
4. **Sentence Transformer** compares semantic similarity with job descriptions  
5. **Regression Model** estimates salary  
6. **Frontend** displays results instantly

---

## 🧪 Example Output

**Input:**  
> “Experienced Python developer with 3 years of experience in machine learning, data analysis, and building AI-driven systems using TensorFlow and Flask.”

**Output:**
Predicted Job Category: Data Science

Top Job Matches:
• Machine Learning — Similarity: 0.672
• Django Developer — Similarity: 0.659
• Software Engineer — Similarity: 0.657

Estimated Salary: ₹700,117

yaml
Copy code

---

## 📁 Project Structure

resume_app/
│
├── app.py # Flask backend
├── templates/
│ └── index.html # Web interface
├── static/
│ ├── style.css # Frontend styling
│ └── script.js # JS logic
├── models/
│ ├── resume_classifier.pkl
│ ├── tfidf_vectorizer.pkl
│ └── salary_predictor.pkl
├── data/
│ ├── job_title_des.csv
│ └── Salary_Dataset_with_Extra_Features.csv
├── requirements.txt
└── README.md

yaml
Copy code

---

💻 Local Setup

**1. Clone Repository**

git clone https://github.com/<yourusername>/resume\_app.git

cd resume\_app



**2. Create and Activate Virtual Environment**

python -m venv venv

\# For Windows

.\\venv\\Scripts\\activate

\# For Mac/Linux

source venv/bin/activate



**3. Install Dependencies**

pip install -r requirements.txt



**4. Run the App**

python app.py



**5. Access Locally**



Visit 👉 http://127.0.0.1:5000/

&nbsp;in your browser.



**🧬 Model Training Summary**

Model	Technique	Purpose

Job Category	TF-IDF + Logistic Regression	Classify resume into job domain

Semantic Matching	SentenceTransformer (all-MiniLM-L6-v2)	Compare resumes and job descriptions

Salary Prediction	Linear Regression	Predict salary based on dataset trends



**🔮 Future Enhancements**



* Add Job Description Input to compare directly with uploaded resumes



* Integrate LLMs (Claude, GPT-4, Gemini) for improved skill extraction



* Build a Resume Scoring System (ATS compatibility, skill gap detection)



* Create Dashboard Analytics for job-market visualization



* Deploy to Render / Vercel / Railway for cloud access



**⚡ Performance Tips**



* Use smaller SentenceTransformer models (e.g., MiniLM) for faster inference

* Cache embeddings for frequently used job descriptions
 
* For large datasets, pre-compute embeddings offline
 
* Run Flask with threaded=True for better concurrency



**📜 License**



This project is licensed under the MIT License — free for personal, academic, or commercial use and modification.



**🤝 Contributing**



Pull requests are welcome!

If you'd like to improve model accuracy or UI/UX, fork the repo and submit a PR.



**👨‍💻 Author**



Vutikuri Shanmukha

AI Developer \& Researcher


&nbsp;| GitHub



**🌟 Star This Repo**



If this project helped you, consider giving it a ⭐ on GitHub — it helps others discover it and supports continued development.

