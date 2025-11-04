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
from typing import Tuple, List, Optional
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


# -------------------- MODEL MANAGER --------------------
class ModelManager:
    """Centralized model management with caching and singleton pattern"""
    
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

    def load_models(self) -> None:
        """Load all models and precompute embeddings"""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return
            
        try:
            logger.info("Loading models...")

            # Load trained classifier
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
            
            # Validate required columns
            if 'Job Description' not in self.job_df.columns:
                raise ValueError("job_title_des.csv must contain 'Job Description' column")
            if 'Job Title' not in self.job_df.columns:
                raise ValueError("job_title_des.csv must contain 'Job Title' column")
                
            logger.info(f"✅ Job dataset loaded: {len(self.job_df)} jobs")

            # Load embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embed_model.max_seq_length = 256
            logger.info("✅ Embedding model loaded")

            # Precompute job embeddings
            self.precompute_job_embeddings()

            self._models_loaded = True
            logger.info("✅ All models loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            logger.error(traceback.format_exc())
            raise

    def precompute_job_embeddings(self) -> None:
        """Precompute and cache job embeddings"""
        try:
            if os.path.exists(EMBEDDING_CACHE_FILE):
                logger.info("Loading cached job embeddings...")
                with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                    self.job_embeddings = pickle.load(f)
                
                # Validate cache size matches dataset
                if len(self.job_embeddings) != len(self.job_df):
                    logger.warning("Cache size mismatch, recomputing embeddings...")
                    self._compute_and_cache_embeddings()
                else:
                    logger.info("✅ Cached job embeddings loaded")
            else:
                self._compute_and_cache_embeddings()
                
        except Exception as e:
            logger.error(f"❌ Error loading cached embeddings: {e}")
            logger.info("Recomputing embeddings...")
            self._compute_and_cache_embeddings()

    def _compute_and_cache_embeddings(self) -> None:
        """Helper to compute and save embeddings"""
        logger.info("Computing new job embeddings...")
        job_descriptions = self.job_df['Job Description'].fillna('').tolist()

        if not job_descriptions:
            raise ValueError("No job descriptions found in dataset")

        self.job_embeddings = self.embed_model.encode(
            job_descriptions,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )

        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(self.job_embeddings, f)

        logger.info("✅ Job embeddings computed and cached")

    def is_loaded(self) -> bool:
        """Check if all models are loaded"""
        return self._models_loaded


# Initialize model manager (singleton)
model_manager = ModelManager()


# -------------------- CONTEXT MANAGERS --------------------
@contextmanager
def temporary_file(file_path: str):
    """Context manager for temporary file cleanup"""
    try:
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")


# -------------------- HELPER FUNCTIONS --------------------
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with improved error handling"""
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), MAX_PDF_PAGES)
            
            if num_pages == 0:
                raise ValueError("PDF has no readable pages")
            
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")
                    continue
        
        extracted_text = text.strip()
        if not extracted_text:
            raise ValueError("No text could be extracted from PDF")
            
        return extracted_text[:MAX_TEXT_LENGTH]
        
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PDF read error: {e}")
        raise ValueError("Invalid or corrupted PDF file")
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise ValueError(f"Unable to extract text from PDF: {str(e)}")


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Handle PDF or TXT extraction with validation"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Text file is empty")
                return content[:MAX_TEXT_LENGTH]
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Text file is empty")
                return content[:MAX_TEXT_LENGTH]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def analyze_resume(resume_text: str) -> Tuple[str, List[Tuple[str, float]], float]:
    """
    Analyze resume and return predictions.
    
    Returns:
        Tuple of (predicted_job, matches, predicted_salary)
    """
    try:
        if not resume_text or len(resume_text.strip()) == 0:
            raise ValueError("Resume text is empty")

        # Ensure models are loaded
        if not model_manager.is_loaded():
            raise RuntimeError("Models not loaded")

        # ---- Predict Job Title ----
        predicted_job = model_manager.resume_classifier.predict([resume_text])[0]
        logger.info(f"Predicted job: {predicted_job}")

        # ---- Semantic Matching ----
        resume_embed = model_manager.embed_model.encode(
            [resume_text],
            convert_to_tensor=True,
            show_progress_bar=False
        )

        cosine_scores = util.cos_sim(resume_embed, model_manager.job_embeddings)
        top_indices = np.argsort(-cosine_scores[0].cpu().numpy())[:TOP_MATCHES]

        matches = []
        for idx in top_indices:
            if idx < len(model_manager.job_df):
                job_title = model_manager.job_df.iloc[idx]['Job Title']
                score = float(cosine_scores[0][idx])
                matches.append((job_title, score))

        # ---- Predict Salary ----
        # Note: Using dummy feature [[10]] - you may want to improve this
        predicted_salary = float(model_manager.salary_model.predict(np.array([[10]]))[0])
        logger.info(f"Predicted salary: ₹{int(predicted_salary):,}")

        return predicted_job, matches, predicted_salary

    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        logger.error(traceback.format_exc())
        raise


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_manager.is_loaded()
    }), 200


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis"""
    try:
        # Validate request
        if 'resume' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Only PDF and TXT files are allowed'
            }), 400

        # Ensure models are loaded
        if not model_manager.is_loaded():
            return jsonify({'error': 'Models not initialized'}), 503

        # Save and process file
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with temporary_file(file_path):
            file.save(file_path)
            logger.info(f"File saved: {filename}")

            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            if not resume_text:
                return jsonify({'error': 'No text extracted from file'}), 400

            # Analyze resume
            predicted_job, matches, salary = analyze_resume(resume_text)

            # Format response
            result = {
                'success': True,
                'predicted_job': predicted_job,
                'matches': [
                    {'title': title, 'score': f"{score:.3f}"}
                    for title, score in matches
                ],
                'salary': f"₹{int(salary):,}"
            }

            return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return jsonify({'error': 'File processing error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Error processing resume. Please try again.'}), 500


@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Direct text prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({'error': 'Text field is empty'}), 400

        if not model_manager.is_loaded():
            return jsonify({'error': 'Models not initialized'}), 503

        # Predict job
        predicted_job = model_manager.resume_classifier.predict([text])[0]
        
        return jsonify({
            'success': True,
            'predicted_job': predicted_job
        }), 200
        
    except Exception as e:
        logger.error(f"Text prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500


# -------------------- ERROR HANDLERS --------------------
@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeded"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


# -------------------- MAIN ENTRY --------------------
if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load models before starting server
    try:
        logger.info("Starting application...")
        model_manager.load_models()
        logger.info("Application ready!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error(traceback.format_exc())
        logger.error("Exiting due to initialization failure")
        exit(1)
    
    # Start Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)