from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create database instance
db = SQLAlchemy()

class BloodTypingResult(db.Model):
    """Model to store blood typing analysis results."""
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(100), nullable=False, unique=True)
    blood_type = db.Column(db.String(5), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sample_label = db.Column(db.String(255), nullable=True)
    image_path = db.Column(db.String(255), nullable=True)
    report_path = db.Column(db.String(255), nullable=True)
    antibody_results = db.Column(db.Text, nullable=True)  # Stored as JSON
    
    def set_antibody_results(self, results):
        """Convert antibody results dictionary to JSON string for storage."""
        if results:
            # Convert complex objects to simpler types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    # Create a copy of the dict to avoid modifying the original
                    new_value = value.copy()
                    # Convert any non-serializable values
                    if 'confidence' in new_value:
                        new_value['confidence'] = float(new_value['confidence'])
                    serializable_results[key] = new_value
                else:
                    serializable_results[key] = value
                    
            self.antibody_results = json.dumps(serializable_results)
    
    def get_antibody_results(self):
        """Convert stored JSON string back to dictionary."""
        if self.antibody_results:
            return json.loads(self.antibody_results)
        return {}