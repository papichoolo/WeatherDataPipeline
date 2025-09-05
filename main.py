from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from extract import fetch_weather_data
from transform import transform_weather_data
from load import save_to_csv, save_to_mongodb, get_weather_data_from_mongodb, list_collections
from config import MONGODB_PASSWORD
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="Weather ETL Pipeline",
    description="Complete ETL pipeline for weather data with MongoDB storage",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Weather ETL Pipeline API",
        "endpoints": [
            "/run-etl - Run complete ETL pipeline",
            "/run-etl-mongodb - Run ETL with MongoDB storage",
            "/weather-data/{collection_name} - Get data from MongoDB collection",
            "/collections - List all MongoDB collections",
            "/health - API health check"
        ]
    }

# @app.get("/run-etl")
# def run_etl():
#     """Run ETL pipeline and save to CSV (original functionality)"""
#     try:
#         raw = fetch_weather_data()
#         if not raw:
#             raise HTTPException(status_code=500, detail="Failed to fetch weather data")
        
#         df = transform_weather_data(raw)
#         filepath = save_to_csv(df)
        
#         return {
#             "status": "ETL completed successfully",
#             "storage": "CSV file",
#             "filepath": filepath,
#             "rows": len(df),
#             "columns": list(df.columns),
#             "cities": df['city'].unique().tolist(),
#             "sample_data": df.head(3).to_dict(orient="records")
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"ETL pipeline failed: {str(e)}")

@app.get("/run-etl-mongodb")
def run_etl_mongodb():
    """Run ETL pipeline and save to MongoDB Atlas"""
    try:
        # Extract weather data
        print("🔄 Starting weather data extraction...")
        raw = fetch_weather_data()
        if not raw:
            raise HTTPException(status_code=500, detail="Failed to fetch weather data")
        
        # Transform data
        print("🔄 Transforming weather data...")
        df = transform_weather_data(raw)
        
        # Save to CSV as backup
        csv_filepath = save_to_csv(df)
        
        # Save to MongoDB
        print("🔄 Saving to MongoDB...")
        mongodb_success = save_to_mongodb(df, MONGODB_PASSWORD)
        
        if not mongodb_success:
            raise HTTPException(status_code=500, detail="Failed to save data to MongoDB")
        
        return {
            "status": "ETL completed successfully",
            "storage": "MongoDB Atlas + CSV backup",
            "csv_filepath": csv_filepath,
            "database": "weather_etl_db",
            "rows": len(df),
            "columns": list(df.columns),
            "cities": df['city'].unique().tolist(),
            "timestamp": datetime.now().isoformat(),
            "collections_created": [
                "raw_weather_data",
                "current_weather", 
                "weather_statistics",
                f"weather_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ],
            "sample_data": df.head(3).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETL pipeline failed: {str(e)}")

@app.get("/weather-data/{collection_name}")
def get_weather_data(collection_name: str):
    """Get weather data from a specific MongoDB collection"""
    try:
        df = get_weather_data_from_mongodb(collection_name, MONGODB_PASSWORD)
        
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")
        
        if df.empty:
            return {
                "collection": collection_name,
                "message": "No data found",
                "count": 0,
                "data": []
            }
        
        return {
            "collection": collection_name,
            "count": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")

@app.get("/collections")
def get_collections():
    """List all available MongoDB collections"""
    try:
        collections = list_collections(MONGODB_PASSWORD)
        return {
            "database": "weather_etl_db",
            "collections": collections,
            "count": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Weather ETL Pipeline"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)