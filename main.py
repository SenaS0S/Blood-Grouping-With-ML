import os
import logging
from web_interface import create_app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create the Flask application - this is needed for Gunicorn
app = create_app()

if __name__ == "__main__":
    # Run the application directly (when not using Gunicorn)
    app.run(host='0.0.0.0', port=5000, debug=True)
