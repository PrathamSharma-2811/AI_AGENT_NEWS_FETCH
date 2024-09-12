from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_agent import LangChainAgent
from config import Config
from flask import Flask, request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_cors import CORS
from dotenv import load_dotenv
import os
from mongo_client import MongoDBClient

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# JWT configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)

# MongoDB configuration
db_client = MongoDBClient(db_url=os.getenv("MONGO_URI"), db_name="newsapp")
if(db_client):
    print("Connected to MongoDB, User can start now!")


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    else:
        result, status = db_client.create_user(username, password)
        return jsonify(result), status
        

    


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if db_client.authenticate_user(username, password):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    # Logout logic can be token blacklisting if needed
    return jsonify({"message": "Logged out successfully"}), 200


@app.route('/home', methods=['GET'])
@jwt_required()
def home():
    current_user = get_jwt_identity()
    return jsonify({"message": f"Welcome {current_user} to the home page!"}), 200

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('query', '')
    api_key = Config.EXTRACT_KEY
    langchain_agent = LangChainAgent(api_key)

    result = langchain_agent.run(question)

    if "output" in result:
        # Directly return the structured JSON response from the agent
        return jsonify(result["output"])
    else:
        return jsonify({"error": result.get("error", "Unknown error occurred")})



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002,debug=True)
