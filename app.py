from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from werkzeug.utils import secure_filename
import PyPDF2
import logging
import pickle
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any
import traceback

# -------------------- CONFIGURATION --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Constants
MAX_TEXT_LENGTH = 5000
MAX_PDF_PAGES = 5
TOP_MATCHES = 3
EMBEDDING_CACHE_FILE = "job_embeddings_cache.pkl"
MIN_TEXT_LENGTH = 50  # Minimum text length for valid resume


# -------------------- MODEL MANAGER --------------------
class ModelManager:
    """Centralized model management with caching and error handling"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.resume_classifier = None
        self.salary_model = None
        self.job_df = None
        self.embed_model = None
        self.job_embeddings = None
        self._models_loaded = False
        self._initialized = True

    def load_models(self):
        """Load all ML models with comprehensive error handling"""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return

        try:
            logger.info("Loading models...")

            # Load classifier
            if not os.path.exists("job_classifier.pkl"):
                raise FileNotFoundError("job_classifier.pkl not found")
            self.resume_classifier = joblib.load("job_classifier.pkl")
            logger.info("✅ Resume classifier loaded")

            # Load salary predictor
            if not os.path.exists("salary_predictor.pkl"):
                raise FileNotFoundError("salary_predictor.pkl not found")
            self.salary_model = joblib.load("salary_predictor.pkl")
            logger.info("✅ Salary predictor loaded")

            # Load job dataset
            if not os.path.exists("job_title_des.csv"):
                raise FileNotFoundError("job_title_des.csv not found")
            self.job_df = pd.read_csv("job_title_des.csv")
            
            # Validate dataset columns
            required_columns = ['Job Description', 'Job Title']
            if not all(col in self.job_df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            
            # Remove any rows with missing critical data
            self.job_df = self.job_df.dropna(subset=required_columns)
            logger.info(f"✅ Job dataset loaded with {len(self.job_df)} entries")

            # Load embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embed_model.max_seq_length = 256
            logger.info("✅ Sentence Transformer model loaded")

            # Precompute embeddings
            self.precompute_job_embeddings()

            self._models_loaded = True
            logger.info("✅ All models successfully initialized")

        except FileNotFoundError as e:
            logger.error(f"Required file missing: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def precompute_job_embeddings(self):
        """Precompute embeddings for job descriptions with validation"""
        try:
            # Try to load cached embeddings
            if os.path.exists(EMBEDDING_CACHE_FILE):
                try:
                    with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                        self.job_embeddings = pickle.load(f)
                    
                    # Validate cache matches current dataset
                    if len(self.job_embeddings) == len(self.job_df):
                        logger.info("✅ Loaded cached job embeddings")
                        return
                    else:
                        logger.warning("Cache size mismatch, recomputing embeddings...")
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, recomputing embeddings...")

            # Compute new embeddings
            job_descriptions = self.job_df['Job Description'].fillna('').tolist()
            
            if not job_descriptions:
                raise ValueError("No job descriptions found in dataset")
            
            logger.info(f"Computing embeddings for {len(job_descriptions)} job descriptions...")
            self.job_embeddings = self.embed_model.encode(
                job_descriptions, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32
            )
            
            # Cache the embeddings
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.job_embeddings, f)
            logger.info("✅ Job embeddings computed and cached")
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            raise

    def is_loaded(self):
        """Check if models are loaded"""
        return self._models_loaded


model_manager = ModelManager()


# -------------------- CONTEXT & HELPERS --------------------
@contextmanager
def temporary_file(file_path):
    """Temporary file cleanup context manager"""
    try:
        yield file_path
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with error handling"""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), MAX_PDF_PAGES)
            
            for i in range(num_pages):
                try:
                    content = reader.pages[i].extract_text()
                    if content:
                        text += content + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    continue
        
        extracted = text.strip()[:MAX_TEXT_LENGTH]
        
        if len(extracted) < MIN_TEXT_LENGTH:
            raise ValueError("Insufficient text extracted from PDF. Please ensure the PDF contains readable text.")
        
        return extracted
        
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from PDF or TXT file with validation"""
    ext = filename.rsplit('.', 1)[-1].lower()
    
    try:
        if ext == 'pdf':
            return extract_text_from_pdf(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()[:MAX_TEXT_LENGTH]
                
            if len(content) < MIN_TEXT_LENGTH:
                raise ValueError("Text file is too short. Please provide a resume with at least 50 characters.")
            
            return content
        else:
            raise ValueError("Unsupported file type")
    except UnicodeDecodeError:
        raise ValueError("Failed to read text file. Please ensure it's a valid UTF-8 encoded text file.")


# -------------------- CORE ANALYSIS --------------------
def analyze_resume(resume_text: str) -> Tuple[str, List[Tuple[str, float]], float]:
    """Predict job, matches, and salary with error handling"""
    try:
        # Validate input
        if not resume_text or len(resume_text.strip()) < MIN_TEXT_LENGTH:
            raise ValueError("Resume text is too short for analysis")
        
        # Predict job category
        predicted_job = model_manager.resume_classifier.predict([resume_text])[0]

        # Generate resume embedding
        resume_embed = model_manager.embed_model.encode(
            [resume_text], 
            convert_to_tensor=True
        )
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(resume_embed, model_manager.job_embeddings)
        top_indices = np.argsort(-cosine_scores[0].cpu().numpy())[:TOP_MATCHES]

        # Get top matches
        matches = []
        for idx in top_indices:
            try:
                job_title = model_manager.job_df.iloc[idx]['Job Title']
                score = float(cosine_scores[0][idx])
                matches.append((job_title, score))
            except Exception as e:
                logger.warning(f"Failed to process match at index {idx}: {e}")
                continue

        # Predict salary (using dummy input for now as per original logic)
        predicted_salary = float(model_manager.salary_model.predict(np.array([[10]]))[0])
        
        return predicted_job, matches, predicted_salary
        
    except Exception as e:
        logger.error(f"Resume analysis failed: {e}")
        raise ValueError(f"Failed to analyze resume: {str(e)}")


def calculate_jd_resume_match(resume_text: str, jd_text: str) -> float:
    """Calculate resume-JD match percentage with validation"""
    try:
        # Validate inputs
        resume_text = resume_text.strip()
        jd_text = jd_text.strip()
        
        if not resume_text or len(resume_text) < MIN_TEXT_LENGTH:
            raise ValueError("Resume text is too short for matching")
        
        if not jd_text or len(jd_text) < 20:
            raise ValueError("Job description is too short for matching")
        
        # Generate embeddings
        resume_embedding = model_manager.embed_model.encode(
            resume_text, 
            convert_to_tensor=True
        )
        jd_embedding = model_manager.embed_model.encode(
            jd_text, 
            convert_to_tensor=True
        )
        
        # Calculate similarity
        similarity_score = util.cos_sim(resume_embedding, jd_embedding).item()
        
        # Convert to percentage and round
        return round(max(0, min(100, similarity_score * 100)), 2)
        
    except Exception as e:
        logger.error(f"JD matching failed: {e}")
        raise ValueError(f"Failed to calculate match: {str(e)}")


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            return jsonify({'error': 'System is still initializing. Please try again.'}), 503

        # Validate file presence
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400

        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Process file
        with temporary_file(file_path):
            file.save(file_path)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Analyze resume
            predicted_job, matches, salary = analyze_resume(resume_text)

            return jsonify({
                'success': True,
                'predicted_job': predicted_job,
                'matches': [{'title': t, 'score': f"{s:.3f}"} for t, s in matches],
                'salary': f"₹{int(salary):,}"
            }), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


@app.route('/match_jd_resume', methods=['POST'])
def match_jd_resume():
    """Calculate resume-JD match percentage"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            return jsonify({'error': 'System is still initializing. Please try again.'}), 503

        # Get inputs
        jd_text = request.form.get('jd_text', '').strip()
        
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        resume_file = request.files['resume']

        if not jd_text:
            return jsonify({'error': 'Please provide a job description'}), 400
        
        if not resume_file or resume_file.filename == '':
            return jsonify({'error': 'Please upload a resume file'}), 400
        
        if not allowed_file(resume_file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400

        # Process file
        filename = secure_filename(resume_file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with temporary_file(file_path):
            resume_file.save(file_path)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Calculate match
            match_percentage = calculate_jd_resume_match(resume_text, jd_text)

            return jsonify({
                'success': True,
                'match_percentage': match_percentage,
                'message': f"The resume matches {match_percentage}% with the job description"
            }), 200

    except ValueError as e:
        logger.warning(f"Validation error in JD match: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"JD Match Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to calculate match percentage. Please try again.'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_manager.is_loaded()
    }), 200


# -------------------- ERROR HANDLERS --------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File size exceeds 16MB limit'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# -------------------- MAIN ENTRY --------------------
if __name__ == '__main__':
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        logger.info("🚀 Initializing AI Resume Analyzer...")
        model_manager.load_models()
        logger.info("✅ System ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000)