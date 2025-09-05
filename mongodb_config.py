import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
from typing import Optional

class MongoDBConfig:
    def __init__(self):
        self.connection_string = "mongodb+srv://tmber:<db_password>@testcluster.wyfpg0v.mongodb.net/?retryWrites=true&w=majority&appName=testcluster"
        self.database_name = "weather_etl_db"
        self.client: Optional[MongoClient] = None
        self.database = None
    
    def connect(self, db_password: str):
        """Connect to MongoDB Atlas"""
        try:
            # Replace password in connection string
            connection_string = self.connection_string.replace("<db_password>", db_password)
            
            # Create client
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            print(f"✅ Successfully connected to MongoDB Atlas!")
            
            # Get database
            self.database = self.client[self.database_name]
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Error connecting to MongoDB: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """Get a specific collection"""
        if self.database is not None:
            return self.database[collection_name]
        return None

# Global MongoDB configuration instance
mongodb_config = MongoDBConfig()
