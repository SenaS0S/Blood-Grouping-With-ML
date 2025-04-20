import os
import cv2
import numpy as np
import base64
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from blood_typing import BloodGroupAnalyzer, validate_image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", "bloodtyping_secret_key")
    
    # Initialize the blood group analyzer
    blood_analyzer = BloodGroupAnalyzer()
    
    # Ensure required directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Configure upload settings
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and process it."""
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Read the file into memory
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Validate the image
            valid, message = validate_image(img)
            if not valid:
                flash(f'Invalid image: {message}', 'error')
                return redirect(request.url)
                
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
            cv2.imwrite(save_path, img)
            
            # Process the image
            try:
                # Analyze the blood sample
                result_data = blood_analyzer.analyze_sample(save_path)
                
                # Generate report image
                report_image, report_path = blood_analyzer.generate_report_image(result_data)
                
                # Store results in session
                session['result_data'] = result_data
                session['report_path'] = report_path
                
                # Redirect to results page
                return redirect(url_for('results'))
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG or JPG image.', 'error')
            return redirect(request.url)
    
    @app.route('/results')
    def results():
        """Display the blood typing results."""
        if 'result_data' not in session:
            flash('No analysis results available. Please upload an image first.', 'error')
            return redirect(url_for('index'))
            
        result_data = session['result_data']
        report_path = session.get('report_path')
        
        # Generate image data for display
        images = {}
        for antibody, data in result_data['antibody_results'].items():
            if data['image_path'] and os.path.exists(data['image_path']):
                # Read the image and encode it as base64 for embedding in HTML
                with open(data['image_path'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    images[antibody] = img_data
        
        # Generate report image data
        report_image = None
        if report_path and os.path.exists(report_path):
            with open(report_path, 'rb') as img_file:
                report_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        return render_template(
            'results.html', 
            result=result_data, 
            images=images,
            report_image=report_image
        )
    
    @app.route('/report/<timestamp>')
    def get_report(timestamp):
        """Serve a specific report image."""
        report_path = os.path.join("results", f"report_{timestamp}.png")
        if os.path.exists(report_path):
            return send_file(report_path, mimetype='image/png')
        else:
            return "Report not found", 404
    
    @app.route('/batch')
    def batch_analysis():
        """Show the batch analysis page."""
        return render_template('batch_analysis.html')
    
    @app.route('/batch/upload', methods=['POST'])
    def batch_upload():
        """Handle batch file upload."""
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files part'})
            
        files = request.files.getlist('files[]')
        labels = request.form.getlist('labels[]') if 'labels[]' in request.form else []
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected files'})
            
        batch_results = []
        
        for i, file in enumerate(files):
            # Get label if available
            label = labels[i] if i < len(labels) else f"Sample {i+1}"
            
            if file and allowed_file(file.filename):
                try:
                    # Read the file into memory
                    file_bytes = file.read()
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Validate the image
                    valid, message = validate_image(img)
                    if not valid:
                        batch_results.append({
                            'filename': file.filename,
                            'label': label,
                            'status': 'error',
                            'message': message
                        })
                        continue
                        
                    # Save the file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
                    cv2.imwrite(save_path, img)
                    
                    # Analyze the sample
                    result_data = blood_analyzer.analyze_sample(save_path)
                    
                    # Generate report
                    _, report_path = blood_analyzer.generate_report_image(result_data)
                    
                    # Add to results
                    batch_results.append({
                        'filename': file.filename,
                        'label': label,
                        'status': 'success',
                        'blood_type': result_data['blood_type'],
                        'confidence': result_data['overall_confidence'],
                        'report_path': os.path.basename(report_path)
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {str(e)}")
                    batch_results.append({
                        'filename': file.filename,
                        'label': label,
                        'status': 'error',
                        'message': str(e)
                    })
            else:
                batch_results.append({
                    'filename': file.filename,
                    'label': label,
                    'status': 'error',
                    'message': 'Invalid file type'
                })
                
        return jsonify({'results': batch_results})
    
    @app.route('/about')
    def about():
        """Show information about the application."""
        return render_template('about.html')
    
    return app
