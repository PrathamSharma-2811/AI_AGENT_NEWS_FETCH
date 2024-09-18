from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import os

# Load environment variables from a .env file
load_dotenv()

# Create a Flask application instance
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Import routes from main.py
from app.main import app

app.config["MONGO_URI"] = os.getenv("MONGO_URI")


# Initialize MongoDB client
mongo = PyMongo(app)


# Register blueprints or modules if you have any
app.register_blueprint(app)
