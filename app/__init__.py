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
from app.main import app as main_app

app.config["MONGO_URI"] = os.getenv("MONGO_URI")


# Initialize MongoDB client
mongo = PyMongo(app)

# Import routes from auth.py
from app.mongo_client import auth_blueprint
app.register_blueprint(auth_blueprint)

# Register blueprints or modules if you have any
# app.register_blueprint(main_app)
