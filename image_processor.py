import cv2
import numpy as np
import os
import logging
from skimage.feature import graycomatrix, graycoprops

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced blood sample image processing class."""
    
    def __init__(self, debug_mode=False):
        """Initialize with debug mode option."""
        self.debug_mode = debug_mode
        os.makedirs("debug", exist_ok=True)
        
    def preprocess(self, image):
        """Enhanced preprocessing pipeline for blood sample images."""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image data")
                
            # Convert to LAB for enhanced processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding for better segmentation
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 4)
            
            # Noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Save debug image if debug mode is enabled
            if self.debug_mode:
                debug_id = np.random.randint(1000)
                cv2.imwrite(f"debug/enhanced_{debug_id}.jpg", enhanced_rgb)
                cv2.imwrite(f"debug/thresh_{debug_id}.jpg", thresh)
                cv2.imwrite(f"debug/cleaned_{debug_id}.jpg", cleaned)
                
            return {
                'original': image,
                'enhanced': enhanced_rgb,
                'grayscale': gray,
                'threshold': thresh,
                'cleaned': cleaned
            }
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise RuntimeError(f"Image preprocessing failed: {str(e)}")
            
    def extract_features(self, processed_images):
        """Extract comprehensive feature set for agglutination detection."""
        try:
            cleaned = processed_images['cleaned']
            original = processed_images['original']
            features = []
            
            # GLCM texture features - crucial for agglutination patterns
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(cleaned, distances=[1, 2], angles=angles,
                              symmetric=True, normed=True)
                              
            # Extract multiple properties from GLCM
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features.extend(graycoprops(glcm, prop).ravel())
                
            # Shape features
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            # If contours found, analyze the largest ones
            if contours:
                # Sort contours by area, descending
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Take the largest contour
                largest = contours[0]
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                
                # Circularity is a good indicator for agglutination
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                
                # Add shape metrics
                features.extend([
                    area, 
                    perimeter, 
                    circularity,
                    len(contours),  # Number of distinct regions
                    sum(cv2.contourArea(c) for c in contours[:3]) / processed_images['original'].size  # Relative area coverage
                ])
                
                # Convexity defects can indicate agglutination patterns
                hull = cv2.convexHull(largest, returnPoints=False)
                if len(hull) > 3:  # Need at least 4 points for convexity defects
                    try:
                        defects = cv2.convexityDefects(largest, hull)
                        if defects is not None:
                            # Count significant defects
                            significant_defects = sum(1 for defect in defects 
                                                  if defect[0][3] > 256)  # Depth threshold
                            features.append(significant_defects)
                        else:
                            features.append(0)
                    except:
                        features.append(0)
                else:
                    features.append(0)
            else:
                # No contours found - likely no agglutination
                features.extend([0, 0, 0, 0, 0, 0])
                
            # Color features - can help differentiate agglutination types
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            for channel in range(3):  # H, S, V channels
                channel_data = hsv[:,:,channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
                
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
            
    def split_sample_image(self, image, num_sections=4):
        """Split a composite blood sample image into individual test sections."""
        try:
            h, w = image.shape[:2]
            
            # Validate dimensions
            if h < 100 or w < 400:
                raise ValueError("Image too small to contain test sections")
                
            sections = []
            
            # Determine orientation based on aspect ratio
            aspect_ratio = w / h
            
            if aspect_ratio > 1.5:  # Horizontal layout
                section_width = w // num_sections
                for i in range(num_sections):
                    section = image[:, i*section_width:(i+1)*section_width]
                    sections.append(section)
            else:  # Vertical layout
                section_height = h // num_sections
                for i in range(num_sections):
                    section = image[i*section_height:(i+1)*section_height, :]
                    sections.append(section)
                    
            # Debug: save sections
            if self.debug_mode:
                for i, section in enumerate(sections):
                    cv2.imwrite(f"debug/section_{i}.jpg", section)
                    
            return sections
            
        except Exception as e:
            logger.error(f"Image splitting failed: {str(e)}")
            raise RuntimeError(f"Could not split image into sections: {str(e)}")
            
    def enhance_visualization(self, image):
        """Enhance the image for better visualization in UI."""
        try:
            # CLAHE enhancement for better visualization
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge enhanced L with original a,b
            merged = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            return image  # Return original if enhancement fails
            
    def overlay_analysis_results(self, image, is_positive, confidence):
        """Overlay agglutination results on the image for visualization."""
        try:
            # Create a copy to not modify original
            result = image.copy()
            
            # Define colors and text based on result
            if is_positive:
                color = (0, 200, 0)  # Green for positive
                text = "POSITIVE"
            else:
                color = (0, 0, 200)  # Red for negative
                text = "NEGATIVE"
                
            # Add colored border
            border_thickness = max(2, int(min(image.shape[0], image.shape[1]) / 50))
            result = cv2.copyMakeBorder(
                result, 
                border_thickness, border_thickness, border_thickness, border_thickness,
                cv2.BORDER_CONSTANT, 
                value=color
            )
            
            # Text specifications
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.7, min(image.shape[0], image.shape[1]) / 500)
            thickness = max(1, int(font_scale * 2))
            
            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (result.shape[1] - text_width) // 2
            text_y = result.shape[0] - border_thickness - 10
            
            # Add text
            cv2.putText(
                result, text, (text_x, text_y), font, font_scale, color, thickness)
                
            # Add confidence percentage
            conf_text = f"{confidence:.1f}%"
            (conf_width, _), _ = cv2.getTextSize(conf_text, font, font_scale * 0.8, thickness - 1)
            conf_x = (result.shape[1] - conf_width) // 2
            conf_y = text_y - text_height - 5
            
            cv2.putText(
                result, conf_text, (conf_x, conf_y), font, font_scale * 0.8, 
                color, thickness - 1)
                
            return result
            
        except Exception as e:
            logger.error(f"Result overlay failed: {str(e)}")
            return image  # Return original if overlay fails
