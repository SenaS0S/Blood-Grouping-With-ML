import cv2
import numpy as np
import os
import logging
from io import BytesIO
import joblib
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
ANTIBODY_TYPES = ["Anti A", "Anti B", "Anti D", "H Antigen Serum Test"]

BLOOD_TYPE_RULES = {
    # Format: (Anti_A, Anti_B, Anti_D, H_Antigen)
    # Standard Types
    (True, False, True, True): "A+",
    (True, False, False, True): "A-",
    (False, True, True, True): "B+",
    (False, True, False, True): "B-",
    (True, True, True, True): "AB+",
    (True, True, False, True): "AB-",
    (False, False, True, True): "O+",
    (False, False, False, True): "O-",
    
    # Bombay Phenotype (H Antigen Negative)
    (True, False, True, False): "Bombay A+",
    (True, False, False, False): "Bombay A-",
    (False, True, True, False): "Bombay B+",
    (False, True, False, False): "Bombay B-",
    (True, True, True, False): "Bombay AB+",
    (True, True, False, False): "Bombay AB-",
    (False, False, True, False): "Bombay O+",
    (False, False, False, False): "Bombay O-"
}

class BloodGroupAnalyzer:
    def __init__(self):
        """Initialize the blood group analyzer."""
        self.models = self.load_models()
        self.setup_directories()
        
    def load_models(self):
        """Load pretrained ML models for each antibody type."""
        models = {}
        try:
            for ab in ANTIBODY_TYPES:
                model_path = f'{ab.replace(" ", "_")}_classifier.pkl'
                if os.path.exists(model_path):
                    models[ab] = joblib.load(model_path)
                    logger.info(f"Loaded classifier for {ab}")
                else:
                    logger.warning(f"Model not found for {ab}: {model_path}")
            
            if models:
                logger.info("Loaded available antibody classifiers")
            else:
                logger.warning("No ML models found. Will use rule-based detection.")
            
            return models
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return None

    def setup_directories(self):
        """Ensure required directories exist."""
        os.makedirs("test", exist_ok=True)
        os.makedirs("debug", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def process_image(self, img):
        """Enhance and process the blood sample image."""
        if img is None or img.size == 0:
            raise ValueError("Invalid image data")
            
        # Convert to LAB for enhanced processing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Adaptive thresholding
        gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 4)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Save debug image
        debug_path = os.path.join("debug", f"processed_{np.random.randint(1000)}.png")
        cv2.imwrite(debug_path, cleaned)
            
        return cleaned

    def extract_features(self, processed_img):
        """Extract comprehensive features for ML classification."""
        features = []
        
        # GLCM texture features
        glcm = graycomatrix(processed_img, distances=[1,2], 
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          symmetric=True, normed=True)
        for prop in ['contrast', 'energy', 'homogeneity', 'correlation']:
            features.extend(graycoprops(glcm, prop).ravel())
        
        # Contour-based shape features
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            features.extend([area, perimeter, circularity])
        else:
            features.extend([0, 0, 0])
            
        # Color features
        hsv = cv2.cvtColor(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        features.append(np.mean(hsv[:,:,1]))  # Average saturation
        
        return np.array(features)

    def analyze_antibody(self, img_path, antibody):
        """Analyze an antibody image with hybrid ML/rule approach."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Invalid image file")
                
            processed = self.process_image(img)
            
            # Save processed image for debugging
            debug_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f"debug/{antibody.replace(' ', '_')}_processed.jpg", debug_img)
            
            # ML-based analysis when available
            if self.models and antibody in self.models:
                features = self.extract_features(processed)
                proba = self.models[antibody].predict_proba([features])[0][1]
                confidence = proba * 100
                agglutination = proba > 0.65
                logger.info(f"{antibody} - ML prediction: {agglutination}, confidence: {confidence:.2f}%")
            else:
                # Rule-based fallback
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
                
                # Dynamic threshold calculation
                agglutination = total_area > (img.size * 0.07)
                
                # Special handling for Anti-D
                if antibody == "Anti D":
                    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
                    energy = graycoprops(glcm, 'energy')[0,0]
                    agglutination = energy < 0.18 if agglutination else energy < 0.35
                
                confidence = 85 if agglutination else 15
                logger.info(f"{antibody} - Rule-based prediction: {agglutination}, confidence: {confidence:.2f}%")
            
            return agglutination, confidence
            
        except Exception as e:
            logger.error(f"Analysis failed for {antibody}: {str(e)}")
            raise RuntimeError(f"Analysis failed for {antibody}: {str(e)}")

    def determine_blood_type(self, results, confidences):
        """Determine blood type from antibody results with confidence scoring."""
        key = tuple(results)
        confidence_avg = sum(confidences) / len(confidences) if confidences else 0
        
        # Check result validity
        blood_type = None
        confidence_note = ""
        
        # Strict match first
        if key in BLOOD_TYPE_RULES:
            blood_type = BLOOD_TYPE_RULES[key]
        else:
            # Find best partial match
            best_match = None
            best_score = 0
            for pattern, bt in BLOOD_TYPE_RULES.items():
                score = sum(a == b for a, b in zip(key, pattern))
                if score > best_score:
                    best_score = score
                    best_match = bt
            
            if best_match and best_score >= 3:
                blood_type = f"{best_match} (Probable)"
            else:
                blood_type = "Undetermined"
        
        # Add confidence information
        if min(confidences) < 60:
            confidence_note = f" (Low confidence: {min(confidences):.1f}%)"
            
        # H Antigen validation
        if ("Bombay" in blood_type and results[3]) or \
           (not "Bombay" in blood_type and not results[3]):
            return "Invalid Result: H Antigen mismatch", 0
            
        return f"{blood_type}{confidence_note}", confidence_avg

    def split_sample_image(self, img_data):
        """Split a multi-antibody sample image into individual sections."""
        try:
            # Convert bytes to image if needed
            if isinstance(img_data, bytes):
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = img_data
                
            h, w = img.shape[:2]
            
            # Validate dimensions
            if h < 100 or w < 400:
                raise ValueError("Image too small to contain test sections")
            
            # Split into sections
            sections = []
            section_paths = {}
            aspect_ratio = w / h
            
            if aspect_ratio > 1.5:  # Horizontal
                section_width = w // 4
                for i, antibody in enumerate(ANTIBODY_TYPES):
                    section = img[:, i*section_width:(i+1)*section_width]
                    sections.append(section)
                    
                    # Save section
                    save_path = os.path.join("test", f"{antibody.replace(' ', '_')}.jpg")
                    cv2.imwrite(save_path, section)
                    section_paths[antibody] = save_path
            else:  # Vertical
                section_height = h // 4
                for i, antibody in enumerate(ANTIBODY_TYPES):
                    section = img[i*section_height:(i+1)*section_height, :]
                    sections.append(section)
                    
                    # Save section
                    save_path = os.path.join("test", f"{antibody.replace(' ', '_')}.jpg")
                    cv2.imwrite(save_path, section)
                    section_paths[antibody] = save_path
            
            return section_paths
            
        except Exception as e:
            logger.error(f"Error splitting sample image: {str(e)}")
            raise ValueError(f"Failed to split sample image: {str(e)}")

    def analyze_sample(self, img_path):
        """Analyze a complete blood sample with 4 antibody tests."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image from {img_path}")
                
            # Split image into sections
            section_paths = self.split_sample_image(img)
            
            # Analyze each section
            results = []
            confidences = []
            antibody_results = {}
            
            for antibody in ANTIBODY_TYPES:
                if antibody in section_paths:
                    agglutination, confidence = self.analyze_antibody(section_paths[antibody], antibody)
                    results.append(agglutination)
                    confidences.append(confidence)
                    antibody_results[antibody] = {
                        "agglutination": agglutination,
                        "confidence": confidence,
                        "image_path": section_paths[antibody]
                    }
                else:
                    results.append(False)
                    confidences.append(0)
                    antibody_results[antibody] = {
                        "agglutination": False,
                        "confidence": 0,
                        "image_path": None
                    }
            
            # Determine blood type
            blood_type, overall_confidence = self.determine_blood_type(results, confidences)
            
            # Generate result data
            result_data = {
                "blood_type": blood_type,
                "overall_confidence": overall_confidence,
                "antibody_results": antibody_results,
                "timestamp": import_datetime_now()
            }
            
            return result_data
            
        except Exception as e:
            logger.error(f"Sample analysis failed: {str(e)}")
            raise RuntimeError(f"Blood sample analysis failed: {str(e)}")
    
    def generate_report_image(self, result_data):
        """Generate a visual report of the blood typing results."""
        try:
            # Create a figure
            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Blood Type: {result_data['blood_type']}", fontsize=16)
            
            # Plot each antibody result
            for i, antibody in enumerate(ANTIBODY_TYPES):
                plt.subplot(2, 2, i+1)
                
                antibody_data = result_data['antibody_results'][antibody]
                if antibody_data['image_path'] and os.path.exists(antibody_data['image_path']):
                    img = cv2.imread(antibody_data['image_path'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                
                result_text = "Positive" if antibody_data['agglutination'] else "Negative"
                confidence = antibody_data['confidence']
                plt.title(f"{antibody}: {result_text} ({confidence:.1f}%)")
                plt.axis('off')
            
            # Add timestamp
            plt.figtext(0.5, 0.01, f"Generated: {result_data['timestamp']}", 
                      ha='center', fontsize=8)
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
            
            # Save to file
            timestamp = result_data['timestamp'].strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join("results", f"report_{timestamp}.png")
            with open(report_path, 'wb') as f:
                f.write(buf.getvalue())
            
            return buf.getvalue(), report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate report: {str(e)}")

def import_datetime_now():
    """Import datetime and return current time."""
    from datetime import datetime
    return datetime.now()

def validate_image(img_data):
    """Quality control checks for input images."""
    try:
        # Convert bytes to image if needed
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = img_data
            
        if img is None:
            return False, "Invalid image file"
            
        if img.shape[0] < 600 or img.shape[1] < 800:
            return False, "Minimum resolution 800x600 required"
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.mean(gray)[0] < 50:
            return False, "Image too dark - adjust lighting"
            
        return True, "Validation passed"
        
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"
