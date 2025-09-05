import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Weather API Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# 50 Cities from different countries for comprehensive weather data
CITIES = [
    # India
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Patna",
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot",
    
    # International Cities
    "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam", "Vienna", "Stockholm",
    "Copenhagen", "Helsinki", "Oslo", "Zurich", "Brussels", "Dublin", "Lisbon", "Prague",
    "Budapest", "Warsaw", "Zagreb", "Athens", "Istanbul", "Moscow", "Kiev", "Minsk",
    "Bucharest", "Sofia", "Belgrade", "Sarajevo", "Skopje", "Tirana"
]

# MongoDB Configuration
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")  # Default password for demo

# Database Collections
COLLECTIONS = {
    "RAW_DATA": "raw_weather_data",
    "CURRENT": "current_weather", 
    "STATISTICS": "weather_statistics",
    "BATCH_PREFIX": "weather_batch_"  # Prefix for batch collections
}

# MLflow & MLOps Configuration
# Use SQLite for local Model Registry support
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "weather-ml")

MODEL_NAMES = {
    "REGRESSOR": os.getenv("MODEL_NAME_REGRESSOR", "weather_temp_regressor"),
    "CLASSIFIER": os.getenv("MODEL_NAME_CLASSIFIER", "weather_condition_classifier"),
}

# Predictions collection for monitoring
PREDICTIONS_COLLECTION = os.getenv("PREDICTIONS_COLLECTION", "predictions")
