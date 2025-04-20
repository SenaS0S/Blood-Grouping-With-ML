import cv2
import numpy as np
import os
import logging
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from image_processor import ImageProcessor
from blood_typing import BloodGroupAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ANTIBODY_TYPES = ["Anti A", "Anti B", "Anti D", "H Antigen Serum Test"]

class ModelEvaluator:
    """Class for evaluating blood type detection models."""
    
    def __init__(self, test_data_path="test_dataset"):
        """Initialize with test dataset path."""
        self.test_data_path = test_data_path
        self.image_processor = ImageProcessor(debug_mode=True)
        self.analyzer = BloodGroupAnalyzer()
        self.load_models()
        
    def load_models(self):
        """Load trained models for evaluation."""
        self.models = {}
        try:
            for ab in ANTIBODY_TYPES:
                model_path = f'{ab.replace(" ", "_")}_classifier.pkl'
                if os.path.exists(model_path):
                    self.models[ab] = joblib.load(model_path)
                    logger.info(f"Loaded model for {ab}")
                else:
                    logger.warning(f"Model not found for {ab}: {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            
    def load_test_data(self):
        """Load test data with ground truth labels."""
        if not os.path.exists(self.test_data_path):
            logger.error(f"Test data directory '{self.test_data_path}' not found!")
            return None
            
        test_data = {ab: {"images": [], "labels": []} for ab in ANTIBODY_TYPES}
        
        # Process each antibody type
        for antibody in ANTIBODY_TYPES:
            antibody_dir = os.path.join(self.test_data_path, antibody.replace(" ", "_"))
            
            if not os.path.exists(antibody_dir):
                logger.warning(f"No test data directory for {antibody}")
                continue
                
            # Look for positive and negative examples
            for label_name in ["positive", "negative"]:
                label_dir = os.path.join(antibody_dir, label_name)
                if not os.path.exists(label_dir):
                    continue
                    
                label_value = 1 if label_name == "positive" else 0
                
                # Load all images in this category
                for img_file in os.listdir(label_dir):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img_path = os.path.join(label_dir, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        logger.warning(f"Could not load {img_path}")
                        continue
                        
                    test_data[antibody]["images"].append(img)
                    test_data[antibody]["labels"].append(label_value)
                    
            logger.info(f"Loaded {len(test_data[antibody]['images'])} test samples for {antibody}")
            
        return test_data
        
    def evaluate_models(self, test_data):
        """Evaluate each model against test data."""
        results = {}
        
        for antibody in ANTIBODY_TYPES:
            if antibody not in self.models:
                logger.warning(f"No model available for {antibody}")
                continue
                
            if antibody not in test_data or len(test_data[antibody]["images"]) == 0:
                logger.warning(f"No test data available for {antibody}")
                continue
                
            logger.info(f"Evaluating model for {antibody}...")
            
            model = self.models[antibody]
            images = test_data[antibody]["images"]
            true_labels = test_data[antibody]["labels"]
            
            # Process each image and extract features
            features = []
            for img in images:
                processed = self.image_processor.preprocess(img)
                feature_vector = self.image_processor.extract_features(processed)
                features.append(feature_vector)
                
            # Make predictions
            pred_labels = model.predict(features)
            
            # Get probabilities for confidence analysis
            pred_probs = model.predict_proba(features)[:, 1]  # Class 1 probabilities
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            report = classification_report(true_labels, pred_labels, output_dict=True)
            cm = confusion_matrix(true_labels, pred_labels)
            
            results[antibody] = {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "pred_probabilities": pred_probs
            }
            
            logger.info(f"{antibody} - Accuracy: {accuracy:.4f}")
            
        return results
        
    def visualize_results(self, evaluation_results):
        """Generate visualization of evaluation results."""
        os.makedirs("evaluation", exist_ok=True)
        
        # Overall summary
        summary = {}
        for antibody, result in evaluation_results.items():
            summary[antibody] = {
                "accuracy": result["accuracy"],
                "precision": result["classification_report"].get(1, {}).get("precision", 0),
                "recall": result["classification_report"].get(1, {}).get("recall", 0),
                "f1": result["classification_report"].get(1, {}).get("f1-score", 0)
            }
            
        # Create summary table
        plt.figure(figsize=(12, 6))
        
        # Extract data for plotting
        antibodies = list(summary.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]
        
        # Group by metric
        data = np.array([[summary[ab][metric] for ab in antibodies] for metric in metrics])
        
        # Create bar chart
        x = np.arange(len(antibodies))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, data[i], width, label=metric.capitalize())
            
        ax.set_ylabel('Scores')
        ax.set_title('Model Evaluation Metrics by Antibody Type')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(antibodies)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("evaluation/model_metrics_summary.png")
        plt.close()
        
        # Individual confusion matrices
        for antibody, result in evaluation_results.items():
            cm = result["confusion_matrix"]
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['Negative', 'Positive'],
                      yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix: {antibody}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"evaluation/{antibody.replace(' ', '_')}_confusion_matrix.png")
            plt.close()
            
            # ROC curve
            if len(np.unique(result["true_labels"])) > 1:  # Only if both classes are present
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(result["true_labels"], result["pred_probabilities"])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: {antibody}')
                plt.legend(loc="lower right")
                plt.savefig(f"evaluation/{antibody.replace(' ', '_')}_roc_curve.png")
                plt.close()
                
            # Prediction confidence distribution
            plt.figure(figsize=(10, 6))
            
            # Separate probabilities by true label
            true_neg = [result["pred_probabilities"][i] for i, l in enumerate(result["true_labels"]) if l == 0]
            true_pos = [result["pred_probabilities"][i] for i, l in enumerate(result["true_labels"]) if l == 1]
            
            plt.hist(true_neg, alpha=0.5, bins=20, range=(0, 1), label='True Negative')
            plt.hist(true_pos, alpha=0.5, bins=20, range=(0, 1), label='True Positive')
            
            plt.xlabel('Predicted Probability of Positive Class')
            plt.ylabel('Frequency')
            plt.title(f'Prediction Confidence Distribution: {antibody}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"evaluation/{antibody.replace(' ', '_')}_confidence_dist.png")
            plt.close()
            
    def evaluate_end_to_end(self, test_samples):
        """Evaluate the complete blood typing pipeline."""
        if not os.path.exists(test_samples):
            logger.error(f"Test samples directory not found: {test_samples}")
            return
            
        results = []
        for sample_file in os.listdir(test_samples):
            if not sample_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            sample_path = os.path.join(test_samples, sample_file)
            
            # Extract ground truth from filename if available
            # Naming convention: bloodtype_sampleID.jpg (e.g., "A+_001.jpg")
            try:
                true_type = sample_file.split('_')[0]
                if true_type not in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]:
                    true_type = "Unknown"
            except:
                true_type = "Unknown"
                
            try:
                # Analyze sample
                result_data = self.analyzer.analyze_sample(sample_path)
                predicted_type = result_data["blood_type"]
                confidence = result_data["overall_confidence"]
                
                # Save result
                results.append({
                    "sample": sample_file,
                    "true_type": true_type,
                    "predicted_type": predicted_type,
                    "confidence": confidence,
                    "correct": true_type in predicted_type if true_type != "Unknown" else None
                })
                
                logger.info(f"Sample: {sample_file} - True: {true_type} - Predicted: {predicted_type} - Confidence: {confidence:.1f}%")
                
            except Exception as e:
                logger.error(f"Error analyzing sample {sample_file}: {str(e)}")
                results.append({
                    "sample": sample_file,
                    "true_type": true_type,
                    "predicted_type": "Error",
                    "confidence": 0,
                    "correct": False
                })
                
        # Summarize results
        correct_count = sum(1 for r in results if r["correct"] is True)
        total_known = sum(1 for r in results if r["correct"] is not None)
        
        if total_known > 0:
            accuracy = correct_count / total_known
            logger.info(f"End-to-end accuracy: {accuracy:.4f} ({correct_count}/{total_known})")
        else:
            logger.info("No samples with known blood types for accuracy evaluation")
            
        # Generate report
        report_path = os.path.join("evaluation", "end_to_end_report.txt")
        with open(report_path, 'w') as f:
            f.write("Blood Type Detection - End-to-End Evaluation Report\n")
            f.write("="*50 + "\n\n")
            
            if total_known > 0:
                f.write(f"Overall Accuracy: {accuracy:.2f} ({correct_count}/{total_known})\n\n")
            else:
                f.write("No samples with known blood types for accuracy evaluation\n\n")
                
            f.write("Sample Results:\n")
            f.write("-"*50 + "\n")
            
            for r in results:
                correct_mark = "✓" if r["correct"] is True else "✗" if r["correct"] is False else "?"
                f.write(f"{r['sample']}: {r['true_type']} → {r['predicted_type']} ({r['confidence']:.1f}%) {correct_mark}\n")
                
        logger.info(f"End-to-end evaluation report saved to {report_path}")
        
        return results
        
    def run_evaluation_pipeline(self):
        """Run the complete evaluation pipeline."""
        logger.info("Starting model evaluation pipeline...")
        
        # Create directories
        os.makedirs("evaluation", exist_ok=True)
        
        # Evaluate individual models
        logger.info("Loading test data...")
        test_data = self.load_test_data()
        
        if test_data:
            logger.info("Evaluating models...")
            evaluation_results = self.evaluate_models(test_data)
            
            logger.info("Generating visualizations...")
            self.visualize_results(evaluation_results)
        else:
            logger.warning("No test data available for model evaluation")
            
        # End-to-end evaluation
        logger.info("Performing end-to-end evaluation...")
        end_to_end_samples = "test_samples"
        if os.path.exists(end_to_end_samples):
            self.evaluate_end_to_end(end_to_end_samples)
        else:
            logger.warning(f"No end-to-end test samples found in {end_to_end_samples}")
            
        logger.info("Evaluation pipeline completed!")
        
def main():
    """Main function to execute evaluation pipeline."""
    evaluator = ModelEvaluator()
    evaluator.run_evaluation_pipeline()

if __name__ == "__main__":
    main()
