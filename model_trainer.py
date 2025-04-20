import cv2
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import matplotlib.pyplot as plt
from image_processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ANTIBODY_TYPES = ["Anti A", "Anti B", "Anti D", "H Antigen Serum Test"]

class BloodTypeModelTrainer:
    """Model trainer for blood type agglutination detection."""
    
    def __init__(self, dataset_path="blood_dataset"):
        """Initialize with dataset path."""
        self.dataset_path = dataset_path
        self.image_processor = ImageProcessor(debug_mode=True)
        self.models = {}
        
    def parse_annotations(self, file_path):
        """Parse YOLO format annotations."""
        annotations = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(f"Invalid annotation line: {line}")
                        continue
                    try:
                        class_label = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        annotations.append((class_label, *coords))
                    except ValueError as e:
                        logger.error(f"Error parsing line: {line} - {e}")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
        return annotations
        
    def load_dataset(self):
        """Load dataset with annotations and extract features."""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset directory '{self.dataset_path}' not found!")
            return None
            
        # Data structures for each antibody type
        antibody_data = {ab: {"features": [], "labels": []} for ab in ANTIBODY_TYPES}
        
        # Initial data exploration counters
        total_images = 0
        valid_annotations = 0
        
        # Track class balance
        class_counts = {ab: {0: 0, 1: 0} for ab in ANTIBODY_TYPES}
        
        # Process each image
        image_files = [f for f in os.listdir(self.dataset_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc="Processing Images"):
            total_images += 1
            
            # Get corresponding annotation file
            base_name = os.path.splitext(image_file)[0]
            annotation_file = os.path.join(self.dataset_path, f"{base_name}.txt")
            if not os.path.exists(annotation_file):
                logger.warning(f"No annotations for {image_file}")
                continue
            
            # Load image
            img = cv2.imread(os.path.join(self.dataset_path, image_file))
            if img is None:
                logger.error(f"Failed to load {image_file}")
                continue
            
            # Load annotations
            annotations = self.parse_annotations(annotation_file)
            if not annotations:
                logger.warning(f"No valid annotations in {annotation_file}")
                continue
                
            # Detect antibody type from filename
            antibody_type = None
            for ab in ANTIBODY_TYPES:
                if ab.replace(" ", "").lower() in image_file.lower():
                    antibody_type = ab
                    break
                    
            if not antibody_type:
                logger.warning(f"Could not determine antibody type for {image_file}")
                continue
                
            # Process each annotation
            for ann in annotations:
                valid_annotations += 1
                class_label, x_center, y_center, width, height = ann
                
                # Convert YOLO coordinates to pixel values
                h, w = img.shape[:2]
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Crop and process the test section
                section = img[y1:y2, x1:x2]
                if section.size == 0:
                    continue
                
                # Process image and extract features
                processed = self.image_processor.preprocess(section)
                feature_vector = self.image_processor.extract_features(processed)
                
                # Add to appropriate dataset
                antibody_data[antibody_type]["features"].append(feature_vector)
                antibody_data[antibody_type]["labels"].append(class_label)
                
                # Update class balance tracker
                class_counts[antibody_type][class_label] += 1
                
        # Log dataset statistics
        logger.info(f"Dataset Summary:")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Valid annotations found: {valid_annotations}")
        
        for ab in ANTIBODY_TYPES:
            neg_count = class_counts[ab][0]
            pos_count = class_counts[ab][1]
            total = neg_count + pos_count
            
            if total > 0:
                logger.info(f"{ab}: {total} samples ({pos_count} positive, {neg_count} negative)")
                logger.info(f"  Class balance: {pos_count/total*100:.1f}% positive")
            else:
                logger.info(f"{ab}: No samples found")
                
        return antibody_data
                
    def train_models(self, antibody_data, test_size=0.2):
        """Train and evaluate models for each antibody type."""
        for antibody in ANTIBODY_TYPES:
            if antibody not in antibody_data:
                logger.warning(f"No data for {antibody}, skipping...")
                continue
                
            features = antibody_data[antibody]["features"]
            labels = antibody_data[antibody]["labels"]
            
            if len(features) < 10:
                logger.warning(f"Insufficient data for {antibody} ({len(features)} samples), skipping...")
                continue
                
            logger.info(f"Training model for {antibody} with {len(features)} samples")
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42)
                
            # Train Random Forest with hyperparameter tuning
            logger.info(f"Training Random Forest for {antibody}...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters for {antibody}: {grid_search.best_params_}")
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test)
            
            logger.info(f"{antibody} - Accuracy: {accuracy:.4f}")
            logger.info(f"{antibody} - Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            model_path = f'{antibody.replace(" ", "_")}_classifier.pkl'
            joblib.dump(best_model, model_path)
            logger.info(f"Model saved as {model_path}")
            
            # Store model reference
            self.models[antibody] = best_model
            
            # Also try SVM if dataset is small
            if len(features) < 50:
                logger.info(f"Small dataset for {antibody}, also trying SVM...")
                
                svm = SVC(probability=True, class_weight='balanced', random_state=42)
                svm.fit(X_train, y_train)
                
                y_pred_svm = svm.predict(X_test)
                accuracy_svm = accuracy_score(y_test, y_pred_svm)
                
                logger.info(f"{antibody} - SVM Accuracy: {accuracy_svm:.4f}")
                
                # If SVM is better, use it instead
                if accuracy_svm > accuracy:
                    logger.info(f"SVM performed better for {antibody}, using it instead")
                    model_path = f'{antibody.replace(" ", "_")}_classifier.pkl'
                    joblib.dump(svm, model_path)
                    self.models[antibody] = svm
                    
    def visualize_feature_importance(self):
        """Visualize feature importance for each Random Forest model."""
        os.makedirs("results", exist_ok=True)
        
        for antibody, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance for {antibody}')
                plt.bar(range(min(20, len(importances))), 
                       importances[indices[:20]], align='center')
                plt.xticks(range(min(20, len(importances))), indices[:20], rotation=90)
                plt.tight_layout()
                plt.savefig(f"results/{antibody.replace(' ', '_')}_feature_importance.png")
                plt.close()
                
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting model training pipeline...")
        
        # Create directories
        os.makedirs("results", exist_ok=True)
        
        # Load and process dataset
        logger.info("Loading dataset...")
        antibody_data = self.load_dataset()
        
        if not antibody_data:
            logger.error("Failed to load dataset!")
            return False
            
        # Train models
        logger.info("Training models...")
        self.train_models(antibody_data)
        
        # Visualize results
        logger.info("Generating visualizations...")
        self.visualize_feature_importance()
        
        logger.info("Training pipeline completed!")
        return True

def main():
    """Main function to execute training pipeline."""
    trainer = BloodTypeModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
