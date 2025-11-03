from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from werkzeug.utils import secure_filename
import PyPDF2
import logging
from pathlib import Path
from functools import lru_cache
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}


class ModelManager:
    """Centralized model management with caching"""
    
    def __init__(self):
        self.resume_classifier = None
        self.tfidf_vectorizer = None
        self.salary_model = None
        self.job_df = None
        self.embed_model = None
        self.job_embeddings = None  # Pre-computed embeddings
        
    def load_models(self):
        """Load all required models and data"""
        try:
            logger.info("Loading models...")
            self.resume_classifier = joblib.load("resume_classifier.pkl")
            self.tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
            self.salary_model = joblib.load("salary_predictor.pkl")
            self.job_df = pd.read_csv("job_title_des.csv")
            
            # Load embedding model with CPU optimization
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embed_model.max_seq_length = 256  # Reduce max sequence length for speed
            
            # Pre-compute job embeddings (THIS IS THE KEY OPTIMIZATION!)
            logger.info("Pre-computing job embeddings...")
            self.precompute_job_embeddings()
            
            logger.info("All models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def precompute_job_embeddings(self):
        """Pre-compute and cache all job description embeddings"""
        embeddings_cache_file = "job_embeddings_cache.pkl"
        
        try:
            # Try to load from cache
            if os.path.exists(embeddings_cache_file):
                logger.info("Loading cached job embeddings...")
                with open(embeddings_cache_file, 'rb') as f:
                    self.job_embeddings = pickle.load(f)
                logger.info("Cached embeddings loaded successfully")
            else:
                # Compute embeddings
                logger.info("Computing job embeddings (this may take a moment)...")
                job_descriptions = self.job_df['Job Description'].tolist()
                
                # Batch encode for better performance
                self.job_embeddings = self.embed_model.encode(
                    job_descriptions,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                    batch_size=32  # Process in batches
                )
                
                # Save to cache
                with open(embeddings_cache_file, 'wb') as f:
                    pickle.dump(self.job_embeddings, f)
                logger.info("Job embeddings computed and cached")
                
        except Exception as e:
            logger.error(f"Error computing job embeddings: {e}")
            raise


# Initialize model manager
model_manager = ModelManager()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(file_path):
    """Extract text content from PDF file - optimized version"""
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), 5)  # Limit to first 5 pages for speed
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Limit text length for faster processing
        return text.strip()[:5000]  # First 5000 characters
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise ValueError("Unable to extract text from PDF")


def extract_text_from_file(file_path, filename):
    """Extract text based on file type"""
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            return text[:5000]  # First 5000 characters for speed
    else:
        raise ValueError("Unsupported file format")


def analyze_resume(resume_text):
    """
    Analyze resume text and return predictions - OPTIMIZED VERSION
    
    Args:
        resume_text: String containing resume content
        
    Returns:
        tuple: (predicted_job, top_matches, predicted_salary)
    """
    try:
        # Validate input
        if not resume_text or len(resume_text.strip()) == 0:
            raise ValueError("Resume text is empty")
        
        # Predict job category (fast operation)
        vectorized_text = model_manager.tfidf_vectorizer.transform([resume_text])
        predicted_job = model_manager.resume_classifier.predict(vectorized_text)[0]
        
        # Find similar job descriptions using PRE-COMPUTED embeddings
        # This is much faster than encoding all job descriptions every time
        resume_embed = model_manager.embed_model.encode(
            [resume_text],
            convert_to_tensor=True,
            show_progress_bar=False  # Disable progress bar for single item
        )
        
        # Calculate cosine similarity with pre-computed embeddings
        cosine_scores = util.cos_sim(resume_embed, model_manager.job_embeddings)
        
        # Get top 3 matches (fast operation)
        top_indices = np.argsort(-cosine_scores[0].cpu().numpy())[:3]
        matches = [
            (
                model_manager.job_df.iloc[idx]['Job Title'],
                float(cosine_scores[0][idx])
            )
            for idx in top_indices
        ]
        
        # Predict salary (fast operation)
        predicted_salary = model_manager.salary_model.predict(np.array([[10]]))[0]
        
        return predicted_job, matches, predicted_salary
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        raise


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis - OPTIMIZED"""
    try:
        # Validate file presence
        if 'resume' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract text from file
            resume_text = extract_text_from_file(file_path, filename)
            
            if not resume_text:
                return jsonify({'error': 'No text could be extracted from the file'}), 400
            
            # Analyze resume (now much faster with pre-computed embeddings)
            predicted_job, matches, salary = analyze_resume(resume_text)
            
            # Format response
            result = {
                'predicted_job': predicted_job,
                'matches': [
                    {
                        'title': title,
                        'score': f"{score:.3f}"
                    }
                    for title, score in matches
                ],
                'salary': f"₹{int(salary):,}"
            }
            
            return jsonify(result)
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {filename}")
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An error occurred while processing your resume'}), 500


@app.errorhandler(413)
def file_too_large(e):
    """Handle file size limit exceeded"""
    return jsonify({'error': 'File size exceeds 16MB limit'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load models before starting server
    try:
        model_manager.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        exit(1)
    
    # Run application
    app.run(debug=True, host='0.0.0.0', port=5000)