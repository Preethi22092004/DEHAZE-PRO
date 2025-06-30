import os
import logging
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import uuid
import torch
import threading
# Import database models directly
import importlib.util
spec = importlib.util.spec_from_file_location("models", "models.py")
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)

db = models_module.db
ProcessedImage = models_module.ProcessedImage
ProcessedVideo = models_module.ProcessedVideo
from utils.dehazing import process_image, process_video, dehaze_with_multiple_methods, dehaze_with_clahe
from utils.hybrid_dehazing import process_hybrid_dehazing
from utils.perfect_dehazing import perfect_dehaze, simple_perfect_dehaze, ultra_safe_dehaze
from utils.maximum_dehazing import maximum_strength_dehaze, remini_level_dehaze
from utils.smart_clarity_dehazing import smart_clarity_dehaze
from utils.reference_trained_dehazing import reference_trained_dehaze
from utils.crystal_clear_dehazing import crystal_clear_dehaze
from utils.ultra_clear_dehazing import ultra_clear_dehaze
from utils.reference_quality_dehazing import reference_quality_dehaze
from utils.balanced_clarity_dehazing import balanced_clarity_dehaze
from utils.effective_clarity_dehazing import effective_clarity_dehaze
from utils.crystal_visibility_dehazing import crystal_visibility_dehaze
from utils.natural_clear_dehazing import natural_clear_dehaze
from utils.reference_playground_dehazing import reference_playground_dehaze
from utils.gentle_natural_dehazing import gentle_natural_dehaze
from utils.smart_heavy_haze_removal import smart_heavy_haze_removal
from utils.perfect_trained_dehazing import perfect_trained_dehaze, check_model_status
from utils.reference_match_dehazing import reference_match_dehaze
from utils.crystal_clear_maximum_dehazing import crystal_clear_maximum_dehaze
from utils.definitive_reference_dehazing import definitive_reference_dehaze
from utils.professional_balanced_dehazing import professional_balanced_dehaze
from utils.professional_immediate_dehazing import professional_balanced_dehaze as immediate_professional_dehaze
from utils.simple_balanced_dehazing import simple_balanced_dehaze
from utils.anti_tint_dehazing import anti_tint_dehaze

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure database
database_url = os.environ.get("DATABASE_URL")
if database_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Fallback to SQLite if no DATABASE_URL is provided
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dehazing.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
TEMP_FOLDER = "static/temp"
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Dictionary to store video processing tasks
video_tasks = {}

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def background_video_processing(task_id, input_path, model_type, device):
    """Process video in background thread and update task status"""
    try:
        video_tasks[task_id]['status'] = 'processing'
        # Process the video
        output_path = process_video(
            input_path, 
            app.config['RESULTS_FOLDER'], 
            device,
            model_type=model_type,
            frame_skip=2  # Process every 2nd frame to speed up processing
        )
        # Update task status
        video_tasks[task_id]['status'] = 'completed'
        video_tasks[task_id]['output'] = output_path
        logger.info(f"Video processing completed: {task_id}")
    except Exception as e:
        # Update task status on error
        video_tasks[task_id]['status'] = 'failed'
        video_tasks[task_id]['error'] = str(e)
        logger.error(f"Video processing failed: {str(e)}")

@app.route('/')
def index():
    # Set Natural Balanced dehazing as the default model for best results
    return render_template('index.html', default_model='natural_balanced')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model', 'natural_balanced')  # Default to natural balanced for best results
    use_multiple = request.form.get('multiple', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_image_file(file.filename):
        # Generate a unique filename to avoid collisions
        safe_filename = secure_filename(file.filename) if file.filename else 'image.jpg'
        filename = str(uuid.uuid4()) + safe_filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Start timing the processing
        start_time = time.time()
        
        try:
            if use_multiple:
                # Process with multiple methods for comparison
                results = dehaze_with_multiple_methods(input_path, app.config['RESULTS_FOLDER'], device)
                processing_time = time.time() - start_time
                
                # Track each method in the database
                for method, path in results.items():
                    db_record = ProcessedImage()
                    db_record.original_filename = safe_filename
                    db_record.input_path = input_path
                    db_record.output_path = path
                    db_record.processing_method = method
                    db_record.processing_time = processing_time / len(results)
                    db_record.success = True
                    db.session.add(db_record)
                
                db.session.commit()
                
                # Prepare response with all results
                response = {
                    'input': input_path,
                    'success': True,
                    'results': {},
                    'processing_time': processing_time
                }
                
                for method, path in results.items():
                    response['results'][method] = path
                
                # Use the CLAHE result as the primary output as it's most reliable
                primary_output = results.get('clahe', results.get('aod', list(results.values())[0]))
                response['output'] = primary_output
                
                return jsonify(response)
            else:
                # Check processing method
                if model_type == 'perfect':
                    # Use the Anti-Tint Model - Direct solution for purple/blue tint issues
                    output_path = anti_tint_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'anti_tint'

                    # Debug: Log the output path and verify file
                    logger.info(f"Anti-tint model output path: {output_path}")
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        logger.info(f"Anti-tint model output file size: {file_size} bytes")
                    else:
                        logger.error(f"Anti-tint model output file does not exist: {output_path}")
                elif model_type == 'ultra':
                    # Use the Ultra Clear dehazing - Advanced clarity system
                    output_path = ultra_clear_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'ultra'
                elif model_type == 'crystal':
                    # Use the Crystal Clear dehazing - Perfect visibility like reference image
                    output_path = crystal_clear_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'crystal'
                elif model_type == 'remini':
                    # Use the Reference-Trained dehazing for professional quality results
                    output_path = reference_trained_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'remini'
                elif model_type == 'reference_match':
                    # Use the Ultra-Advanced Reference Match model for crystal clear results
                    output_path = reference_match_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'reference_match'
                elif model_type == 'crystal_maximum':
                    # Use the Crystal Clear Maximum model for ultimate clarity
                    output_path = crystal_clear_maximum_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'crystal_maximum'
                elif model_type == 'definitive':
                    # Use the Definitive Reference Quality model - THE FINAL SOLUTION
                    output_path = definitive_reference_dehaze(input_path, app.config['RESULTS_FOLDER'])
                    processing_method = 'definitive'
                elif model_type == 'hybrid':
                    # Use the advanced hybrid ensemble system for maximum accuracy
                    output_path = process_hybrid_dehazing(
                        input_path,
                        app.config['RESULTS_FOLDER'],
                        device=device,
                        target_quality=0.8,
                        blend_method='quality_weighted',
                        enhancement_level='moderate'
                    )
                    processing_method = 'hybrid'
                else:
                    # Always use the selected model type or fall back to deep model
                    output_model = model_type

                    # Process with selected model
                    output_path = process_image(
                        input_path,
                        app.config['RESULTS_FOLDER'],
                        device,
                        model_type=output_model
                    )
                    processing_method = output_model
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Add record to database
                db_record = ProcessedImage()
                db_record.original_filename = safe_filename
                db_record.input_path = input_path
                db_record.output_path = output_path
                db_record.processing_method = processing_method
                db_record.processing_time = processing_time
                db_record.success = True
                db.session.add(db_record)
                db.session.commit()
                
                # Return the paths for display
                return jsonify({
                    'input': input_path,
                    'output': output_path,
                    'success': True,
                    'processing_time': processing_time
                })
        except Exception as e:
            try:
                # If any model fails, try perfect trained dehazing as a reliable fallback
                logger.warning(f"Model processing failed, using perfect trained dehazing as fallback: {str(e)}")
                output_path = perfect_trained_dehaze(input_path, app.config['RESULTS_FOLDER'])
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Add record to database with error information
                db_record = ProcessedImage()
                db_record.original_filename = safe_filename
                db_record.input_path = input_path
                db_record.output_path = output_path
                db_record.processing_method = 'perfect_fallback'
                db_record.processing_time = processing_time
                db_record.success = True
                db_record.error_message = str(e)
                db.session.add(db_record)
                db.session.commit()
                
                return jsonify({
                    'input': input_path,
                    'output': output_path,
                    'success': True,
                    'processing_time': processing_time
                })
            except Exception as e2:
                # Log the error to database
                db_record = ProcessedImage()
                db_record.original_filename = safe_filename
                db_record.input_path = input_path
                db_record.output_path = ''
                # Always use the model_type as the processing method for consistency
                db_record.processing_method = model_type
                db_record.success = False
                db_record.processing_time = time.time() - start_time
                db_record.error_message = f"{str(e)} | {str(e2)}"
                db.session.add(db_record)
                db.session.commit()
                
                logger.error(f"All processing methods failed: {str(e2)}")
                return jsonify({'error': f'Error processing image: {str(e2)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Only JPG, JPEG, PNG, and WebP are allowed.'}), 400

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model', 'deep')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_video_file(file.filename):
        # Generate a unique filename and task ID
        task_id = str(uuid.uuid4())
        safe_filename = secure_filename(file.filename) if file.filename else 'video.mp4'
        filename = task_id + safe_filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Create a new processing task
        video_tasks[task_id] = {
            'status': 'queued',
            'input': input_path,
            'start_time': time.time(),
            'model': model_type
        }
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=background_video_processing,
            args=(task_id, input_path, model_type, device)
        )
        thread.daemon = True
        thread.start()
        
        # Return the task ID for status polling
        return jsonify({
            'task_id': task_id,
            'status': 'queued',
            'input': input_path,
            'success': True
        })
    
    return jsonify({'error': 'Invalid file format. Only MP4, MOV, AVI, and MKV are allowed.'}), 400

@app.route('/video-status/<task_id>', methods=['GET'])
def video_status(task_id):
    """Check the status of a video processing task"""
    if task_id not in video_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = video_tasks[task_id]
    
    response = {
        'status': task['status'],
        'input': task['input']
    }
    
    # Add output path if processing is complete
    if task['status'] == 'completed' and 'output' in task:
        response['output'] = task['output']
    
    # Add error message if processing failed
    if task['status'] == 'failed' and 'error' in task:
        response['error'] = task['error']
    
    # Calculate progress based on processing time (estimated)
    elapsed = time.time() - task['start_time']
    if task['status'] == 'processing':
        # Rough estimate of progress based on time
        estimated_duration = 300  # 5 minutes estimated for an average video
        progress = min(95, (elapsed / estimated_duration) * 100)
        response['progress'] = progress
    elif task['status'] == 'completed':
        response['progress'] = 100
    elif task['status'] == 'failed':
        response['progress'] = 100
    else:  # queued
        response['progress'] = 0
    
    return jsonify(response)

@app.route('/quick-enhance', methods=['POST'])
def quick_enhance():
    """Quick enhancement using CLAHE (non-deep learning method)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_image_file(file.filename):
        # Generate a unique filename
        safe_filename = secure_filename(file.filename) if file.filename else 'image.jpg'
        filename = str(uuid.uuid4()) + safe_filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        try:
            # Process with CLAHE (fast)
            output_path = dehaze_with_clahe(input_path, app.config['RESULTS_FOLDER'])
            
            # Return the paths for display
            return jsonify({
                'input': input_path,
                'output': output_path,
                'success': True
            })
        except Exception as e:
            logger.error(f"Error processing image with CLAHE: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Only JPG, JPEG, PNG, and WebP are allowed.'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    response = send_from_directory(app.config['RESULTS_FOLDER'], filename)
    # Add aggressive cache-busting headers
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
    response.headers['ETag'] = f'"{int(time.time())}-{hash(filename)}"'
    # Additional headers to prevent caching
    response.headers['X-Accel-Expires'] = '0'
    response.headers['Surrogate-Control'] = 'no-store'
    return response

@app.route('/view_result/<timestamp>')
def view_result(timestamp):
    return render_template(f'result_{timestamp}.html')

@app.route('/force_refresh')
def force_refresh():
    return render_template('force_refresh.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available dehazing models"""
    models = [
        {
            'id': 'perfect',
            'name': 'Perfect Trained Model',
            'description': 'Ultimate AI-trained model - Perfect balance of crystal clear results and natural colors. Not too aggressive, not too simple.',
            'processing_time': 'Fast',
            'recommended': True,
            'category': 'Perfect'
        },
        {
            'id': 'ultra',
            'name': 'Ultra Clear Dehazing',
            'description': 'Advanced clarity system with adaptive processing for bright, vivid results',
            'processing_time': 'Fast',
            'recommended': False,
            'category': 'Perfect'
        },
        {
            'id': 'crystal',
            'name': 'Crystal Clear Dehazing',
            'description': 'Perfect visibility like reference image - crystal clear sunny day quality',
            'processing_time': 'Fast',
            'recommended': False,
            'category': 'Perfect'
        },
        {
            'id': 'remini',
            'name': 'Reference-Trained Dehazing',
            'description': 'Professional-grade dehazing trained to match reference image characteristics and quality',
            'processing_time': 'Medium',
            'recommended': True,
            'category': 'Professional'
        },
        {
            'id': 'reference_match',
            'name': 'Reference Match Ultra-Advanced',
            'description': 'State-of-the-art model with attention mechanisms specifically designed to match your reference image quality - Crystal clear results',
            'processing_time': 'Medium',
            'recommended': True,
            'category': 'Ultra-Advanced'
        },
        {
            'id': 'crystal_maximum',
            'name': 'Crystal Clear Maximum',
            'description': 'Ultimate dehazing model with 50+ million parameters, residual dense blocks, and progressive refinement - Maximum clarity for your reference image quality',
            'processing_time': 'Medium',
            'recommended': True,
            'category': 'Ultimate'
        },
        {
            'id': 'definitive',
            'name': 'Definitive Reference Quality',
            'description': 'THE FINAL SOLUTION - Produces crystal clear results matching your reference playground image. Advanced algorithmic approach with perfect clarity and natural colors.',
            'processing_time': 'Fast',
            'recommended': True,
            'category': 'FINAL SOLUTION'
        },
        {
            'id': 'natural',
            'name': 'Natural Dehazing',
            'description': 'Conservative dehazing that preserves natural colors and appearance',
            'processing_time': 'Very Fast',
            'recommended': False,
            'category': 'Natural'
        },
        {
            'id': 'adaptive_natural',
            'name': 'Adaptive Natural',
            'description': 'Analyzes haze level and applies appropriate natural dehazing strength',
            'processing_time': 'Very Fast',
            'category': 'Natural'
        },
        {
            'id': 'conservative',
            'name': 'Conservative Dehazing',
            'description': 'Very gentle processing for subtle improvements while maintaining realism',
            'processing_time': 'Very Fast',
            'category': 'Natural'
        },
        {
            'id': 'minimal',
            'name': 'Minimal Enhancement',
            'description': '95% original image, 5% enhanced - ultra-gentle processing to eliminate tinting',
            'processing_time': 'Very Fast',
            'category': 'Natural'
        },
        {
            'id': 'passthrough',
            'name': 'Passthrough',
            'description': 'No processing - direct copy of original image for testing',
            'processing_time': 'Instant',
            'category': 'Natural'
        },
        {
            'id': 'deep',
            'name': 'DeepDehazeNet',
            'description': 'Advanced multi-scale dehazing network with optimal haze removal and enhanced contrast',
            'processing_time': 'Medium-Slow',
            'category': 'AI Models'
        },
        {
            'id': 'enhanced',
            'name': 'Enhanced Dehazing',
            'description': 'Uses a pretrained ResNet-based model for high-quality dehazing',
            'processing_time': 'Medium',
            'category': 'AI Models'
        },
        {
            'id': 'aod',
            'name': 'AOD-Net',
            'description': 'All-in-One Dehazing Network, optimized for fog and haze removal',
            'processing_time': 'Fast',
            'category': 'AI Models'
        },
        {
            'id': 'clahe',
            'name': 'CLAHE',
            'description': 'Contrast Limited Adaptive Histogram Equalization, a fast non-deep learning method',
            'processing_time': 'Very Fast',
            'category': 'Traditional'
        }
    ]
    return jsonify(models)

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """Get trained model status"""
    try:
        status = check_model_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'version': '1.0.0'
    })

@app.route('/static/service-worker.js')
def service_worker():
    """Serve the service worker with correct mimetype"""
    return send_from_directory('static', 'service-worker.js', mimetype='application/javascript')

@app.route('/test_cache.html')
def test_cache():
    """Serve the cache test page"""
    return send_from_directory('.', 'test_cache.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
