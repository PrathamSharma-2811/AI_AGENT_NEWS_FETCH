from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

class MongoDBClient:
    def __init__(self, db_url, db_name):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]

    def create_user(self, username, password):
        if self.db.users.find_one({"username": username}):
            return {"error": "User already exists"}, 400

        hashed_password = generate_password_hash(password)
        self.db.users.insert_one({"username": username, "password": hashed_password})
        return {"message": "User created successfully"}, 201

    def authenticate_user(self, username, password):
        user = self.db.users.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            return True
        return False

    def get_user(self, username):
        return self.db.users.find_one({"username": username})
