# AI Resume Analyzer

An AI-powered web application that **analyzes resumes**, **matches them with job descriptions**, and **predicts the best-fit job titles and estimated salaries** with advanced NLP and machine learning.

-----

##  Features

This application leverages advanced NLP to provide comprehensive insights into job fit and career potential.

### Core Features

  * ✅ **Resume Upload** — Supports **PDF and TXT** formats for easy resume submission.
  * ✅ **Job Description Analysis** — Input any job description (JD) to check compatibility with detailed breakdowns.
  * ✅ **AI-Powered Matching** — Uses state-of-the-art NLP models for core predictions:
      * **Predicted Job Title** that best fits your profile.
      * **Skill Match Percentage** between your resume and the JD.
      * **Estimated Salary Range** with confidence indicators.
  * ✅ **Modern UI/UX** — Features a stunning **Dark Theme** with **Glassmorphism** effects, smooth animations, and a responsive design.
  * ✅ **Real-Time Insights** — Receive instantaneous analysis and results upon submission without page reloads.

### 🆕 Phase 1 Enhancements (Latest)

  *  **Advanced Skills Extraction** — Detects **200+ technical skills** across 13 categories with intelligent variation handling (e.g., "Node" → "Node.js", "K8s" → "Kubernetes")
  *  **Missing Keywords Detection** — Identifies keywords from job descriptions that are missing from your resume, ranked by importance:
      *  **Critical** (High Priority)
      *  **Important** (Medium Priority)
      *  **Optional** (Low Priority)
  *  **Actionable Recommendations** — Provides specific suggestions on which keywords and skills to add
  *  **Skills Breakdown** — Categorized view of matched vs. missing skills by technology domain
  *  **Feature-Based Salary Prediction** — Uses extracted resume features (experience, education, seniority, skills) instead of dummy data
  *  **Confidence Scoring** — Transparent confidence indicators showing prediction reliability
  *  **Analytics Tracking** — Logs usage patterns and model performance for continuous improvement
  * ⏱️ **Rate Limiting** — Prevents abuse with 10 requests/minute protection

-----

##  Tech Stack

| Category | Technologies | Description |
| :--- | :--- | :--- |
| **Backend** | **Flask (Python)** | Robust and lightweight web framework for the backend. |
| **Frontend** | HTML5, CSS3, JavaScript | Modern, responsive UI with Glassmorphism and animations. |
| **AI/NLP Models** | **Sentence Transformers, Scikit-learn** | Core components for text embedding, similarity, and job title prediction. |
| **Libraries Used** | PyPDF2, NumPy, Pandas, Joblib, Flask-Limiter | PDF parsing, data manipulation, model serialization, and rate limiting. |
| **Deployment** | Render / Railway / AWS | Recommended platforms for production deployment. |

-----

## ⚙️ Local Setup

Follow these steps to get the **AI Resume Analyzer** running on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2️⃣ Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

All required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

Start the Flask application server.

```bash
python app.py
```

Or to suppress warnings:

```bash
python -W ignore app.py
```

### 5️⃣ Open in Browser

The application will be accessible locally.

```bash
Visit  http://127.0.0.1:5000/
```

-----

##  Project Structure

```
AI-Resume-Analyzer/
│
├── app.py                      # Main Flask backend file
├── config.py                   # Configuration settings
├── skills_taxonomy.json        # Comprehensive skills database (200+ skills)
├── analytics.json              # Usage analytics and tracking data
├── job_classifier.pkl          # Trained model for job prediction
├── salary_predictor.pkl        # Trained model for salary estimation
├── job_title_des.csv           # Dataset for job titles and descriptions
│
├── static/
│   ├── style.css               # Modern Dark Theme Styling (1400+ lines)
│   └── script.js               # Frontend logic and API handling
│
├── templates/
│   └── index.html              # Single-page application interface
│
├── uploads/                    # Temporary storage for uploaded resumes
├── requirements.txt
└── README.md
```

-----

##  How It Works

### Resume Analysis Flow

1.  **Input:** User uploads a **resume (PDF or TXT)** and optionally provides a **job description (JD)**.
2.  **Extraction:** Text is extracted from the file using **PyPDF2** (for PDFs) or standard text reading.
3.  **Feature Extraction:** The system extracts key features from the resume:
    - Years of experience (from text patterns and date ranges)
    - Education level (Bachelor's, Master's, PhD)
    - Seniority level (Entry, Mid, Senior, Lead)
    - Technical skills (200+ skills across 13 categories)
4.  **Embedding:** The **SentenceTransformer** model converts the resume and JD text into numerical vector **embeddings**.
5.  **Matching:** **Cosine similarity** is calculated between the embeddings to determine the **Skill Match Percentage**.
6.  **Prediction:** A trained **Machine Learning model** predicts the best-fit **Job Title** and **Estimated Salary**.
7.  **Insights:** The system provides **detailed breakdowns**, **missing keywords**, and **smart suggestions** for resume improvement.

###  Confidence Level Calculation

The **Salary Prediction Confidence Score** is calculated based on **feature completeness** — how many resume features were successfully extracted.

#### Calculation Formula

```
Confidence Score = (Number of Features Found / Total Features) × 100
```

#### Features Checked (4 Total)

1. **Years of Experience** — Did we extract experience duration?
   - Patterns: "5 years experience", "2019-2023"
   - Weight: ×2.0 (highest impact on salary)

2. **Education Level** — Did we find a degree?
   - Levels: 0=Unknown, 1=Bachelor's, 2=Master's, 3=PhD
   - Weight: ×3.0 (very high impact)

3. **Seniority Level** — Did we detect job level?
   - Levels: 0=Entry, 1=Mid, 2=Senior, 3=Lead/Principal
   - Weight: ×2.5 (high impact)

4. **Skills Count** — Did we find technical skills?
   - Detected from 200+ skill taxonomy
   - Weight: ×0.5 (moderate impact)

#### Confidence Levels

| Score | Badge Color | Meaning |
|-------|-------------|---------|
| **≥80%** |  Green (High) | All or most features found — reliable prediction |
| **50-79%** |  Yellow (Medium) | Some features missing — moderate reliability |
| **<50%** |  Red (Low) | Many features missing — low reliability |

#### Example Calculations

**High Confidence (100%):**
```
Resume: "5 years experience, Master's degree, Senior Developer, Python, AWS, Docker"
✓ Experience: 5 years
✓ Education: Master's (level 2)
✓ Seniority: Senior (level 2)
✓ Skills: 3+ skills found
Result: 4/4 = 100% confidence
```

**Medium Confidence (75%):**
```
Resume: "Bachelor's degree, Software Engineer, Java, Spring"
✗ Experience: Not found
✓ Education: Bachelor's (level 1)
✓ Seniority: Mid (level 1)
✓ Skills: 2+ skills found
Result: 3/4 = 75% confidence
```

**Low Confidence (25%):**
```
Resume: "JavaScript developer"
✗ Experience: Not found
✗ Education: Not found
✗ Seniority: Not found
✓ Skills: 1 skill found
Result: 1/4 = 25% confidence
```

#### Salary Calculation

The extracted features are combined using weighted values:

```python
Salary Input = (years_experience × 2.0) + 
               (education_level × 3.0) + 
               (seniority_level × 2.5) + 
               (skills_count × 0.5)
```

This combined value is then passed to the pre-trained ML model for salary prediction.

> **⚠️ Important Note:** The confidence score reflects **data completeness**, not prediction accuracy. The actual salary prediction depends on the quality of the trained model. The current model was trained with limited data and should be retrained with real salary datasets for production use.

-----

###  Example Output

| Metric | Result |
| :--- | :--- |
| **Predicted Job Title** | Data Scientist |
| **Match Score** | **86%** |
| **Estimated Salary** | ₹8.2 – ₹9.5 LPA |
| **Confidence** |  High (100%) |
| **Missing Keywords** |  Deep Learning, TensorFlow  MLOps |
| **Skills Breakdown** | ✅ Python, Pandas, NumPy ❌ PyTorch, Keras |
| **Suggestions** | Add critical keywords: "Deep Learning", "Model Optimization" |

-----

##  Skills Taxonomy

The application uses a comprehensive **skills taxonomy** with **200+ technical skills** organized into **13 categories**:

-  **Programming Languages** (Python, Java, JavaScript, C++, etc.)
-  **Web Frameworks** (React, Angular, Node.js, Django, etc.)
- ️ **Databases** (MySQL, PostgreSQL, MongoDB, Redis, etc.)
- ☁️ **Cloud Platforms** (AWS, Azure, GCP)
-  **DevOps Tools** (Docker, Kubernetes, Jenkins, Terraform, etc.)
-  **Data Science & ML** (TensorFlow, PyTorch, Scikit-learn, etc.)
-  **Mobile Development** (React Native, Flutter, Swift, Kotlin)
-  **Testing Frameworks** (Jest, Pytest, Selenium, etc.)
- ⚙️ **Other Technologies** (REST API, GraphQL, Microservices, etc.)
-  **Methodologies** (Agile, Scrum, DevOps, CI/CD)
-  **Soft Skills** (Leadership, Communication, Problem Solving)
-  **Design Tools** (Figma, Adobe XD, Sketch)
- ️ **Other Tools** (Git, JIRA, Postman, etc.)

Each skill supports **multiple variations** for robust detection (e.g., "Node" → "Node.js", "K8s" → "Kubernetes").

-----

##  Analytics & Tracking

The application tracks usage patterns and model performance in `analytics.json`:

- **Total Uploads** — Number of resumes analyzed
- **Total Matches** — Number of JD match requests
- **Average Match Score** — Mean compatibility percentage
- **Prediction Logs** — Individual analysis results with timestamps

This data helps improve the models and understand user patterns.

-----

##  Rate Limiting

To prevent abuse and ensure fair usage, the application implements rate limiting:

- **10 requests per minute** per endpoint
- **50 requests per hour** (default)
- **200 requests per day** (default)

When the limit is exceeded, users receive a friendly error message with a countdown timer.

-----

##  Future Enhancements

We are always looking to improve! Potential future features include:

  * ️ Voice Resume Input for accessibility.
  *  Integration with large language models (OpenAI / Gemini) for detailed resume rewrites.
  *  Multi-language resume parsing capabilities.
  *  Data visualization of career insights and trends.
  *  Auto-suggest and implement resume corrections.
  *  ATS optimization scoring and recommendations.
  *  Analytics dashboard for administrators.
  *  Resume comparison mode for multiple candidates.

-----

## ‍ Author

Developed by: **Vutikuri Shanmukha**

  *  B.Tech in Electronics & Communication Engineering
  *  Passionate about AI, NLP, and Human–Machine synergy

-----

##  License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for any purpose.

-----

##  Acknowledgements

A special thanks to the following for their contributions and resources:

  * **Sentence Transformers Team** for the powerful NLP models.
  * **Flask Community** for the robust and versatile web framework.
  * **Scikit-learn Contributors** for the comprehensive ML library.
  * **OpenAI** for inspiration on AI-based resume screening solutions.

-----

##  Version History

### v2.0.0 - Phase 1 Enhancements (Latest)
- ✅ Advanced skills extraction with 200+ skills taxonomy
- ✅ Missing keywords detection with importance ranking
- ✅ Feature-based salary prediction with confidence scoring
- ✅ Analytics tracking and usage monitoring
- ✅ Rate limiting for API protection
- ✅ Enhanced UI with color-coded badges and breakdowns

### v1.0.0 - Initial Release
- ✅ Basic resume analysis and job matching
- ✅ Salary prediction (dummy data)
- ✅ Simple skills extraction
- ✅ Dark theme UI

-----

**Made with ❤️ using Flask, Python, and AI**
