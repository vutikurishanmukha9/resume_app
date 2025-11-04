# 🧠 Resume Analyzer – AI-Powered Resume Intelligence (resume_app)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-driven web application that analyzes resumes, predicts ideal job roles, matches job descriptions, and estimates salary ranges — all powered by machine learning and natural language processing (NLP).
Think of it as your personal career analyst — intelligent, fast, and open-source.

**🚀 Overview**

Resume Analyzer bridges the gap between your skills and the job market using data-driven intelligence.
Built with Flask, scikit-learn, and Sentence Transformers, it performs three key functions:

🧩 Predicts your most likely job category from resume text or PDF.

🔍 Finds semantically similar job descriptions using NLP embeddings.

💰 Estimates an expected salary using regression-based modeling.

It’s simple, interpretable, and lightning-fast — perfect for students, recruiters, and career platforms.

**🧠 Core Features**

✅ Smart Resume Understanding – Extracts meaningful information (skills, roles, tools) from resume text or PDFs.
✅ Job Role Prediction – Uses trained ML models to classify resumes into the right job category.
✅ Semantic Job Matching – Compares your resume with real-world job descriptions using all-MiniLM-L6-v2.
✅ Salary Estimation – Predicts salary ranges based on skills and domain trends.
✅ Web Interface – Clean, responsive UI built with HTML, CSS, and JS.
✅ Customizable Models – Retrain or fine-tune models for specific industries or geographies.

**⚙️ Architecture**
User → Resume Upload → Text Extraction → TF-IDF Vectorization
     → Job Category Prediction → Semantic Similarity Matching
     → Salary Estimation → Result Visualization (Frontend)

**🧩 Tech Stack**
Layer	Technology
Frontend	HTML5, CSS3, JavaScript
Backend	Flask (Python)
ML/NLP Models	scikit-learn, Sentence Transformers
Vectorization	TF-IDF
Embedding Model	all-MiniLM-L6-v2
PDF Parsing	PyPDF2
Data Handling	pandas, numpy
Persistence	joblib
Version Control	Git + GitHub
📊 Example Output

**Input:**

“Python developer with 3+ years of experience in ML, data analysis, and Flask-based applications.”

Output:

Prediction Type	Result
Job Category	Data Scientist
Top Matches	Machine Learning Engineer (0.67), Python Developer (0.66), Software Engineer (0.64)
Estimated Salary	₹7,00,000 per annum


📁 Project Structure
resume_app/
├── app.py                        # Flask backend
├── templates/
│   └── index.html                # Main web interface
├── static/
│   ├── style.css                 # Frontend styling
│   └── script.js                 # Frontend logic
├── models/
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── salary_predictor.pkl
├── data/
│   ├── job_title_des.csv
│   └── Salary_Dataset_with_Extra_Features.csv
├── requirements.txt
└── README.md

**💻 Local Setup**

1. Clone the Repository

git clone https://github.com/<yourusername>/resume_app.git

cd resume_app


**2. Create and Activate Virtual Environment**

python -m venv venv
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate


**3. Install Dependencies**

pip install -r requirements.txt


**4. Run the App**

python app.py


**5. Open in Browser**

http://127.0.0.1:5000

**🧬 Model Summary**
Model	Technique	Purpose
Job Classifier	TF-IDF + Logistic Regression	Classify resume into job domain
Semantic Matcher	SentenceTransformer (MiniLM-L6-v2)	Compare resumes & job descriptions
Salary Predictor	Linear Regression	Estimate salary based on skillset


**🔮 Future Enhancements**

*🚧 Next Planned Features:*

LLM integration (GPT-4, Claude, Gemini) for advanced skill and gap extraction

Resume scoring & ATS compatibility report

Dashboard with analytics and salary visualization

Cloud deployment (Render, Railway, or Vercel)

CSV export for resume analysis reports

**⚡ Performance Tips**

Use smaller embedding models for faster inference.

Cache JD embeddings to avoid recomputation.

Pre-compute job description vectors for large datasets.

Run Flask with threaded=True for better concurrency.

**📜 License**

This project is licensed under the MIT License — free for personal, academic, and commercial use.

**🤝 Contributing**

Contributions are always welcome!
If you have ideas to improve model accuracy, UX, or performance — fork the repo, make changes, and open a PR.

**👨‍💻 Author**

Vutikuri Shanmukha
AI Developer & Researcher

**🌟 Support**

If you find this project useful, please star ⭐ the repository —
it helps others discover it and supports ongoing development!
