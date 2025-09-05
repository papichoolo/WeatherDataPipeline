from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from extract import fetch_weather_data
from transform import transform_weather_data
from load import save_to_csv, save_to_mongodb, get_weather_data_from_mongodb, list_collections
from config import MONGODB_PASSWORD
import pandas as pd
from datetime import datetime
import os

# ML imports
from ml.training import retrain_from_mongo
from ml.predict import predict, evaluate_and_log, evaluate_with_details
from ml.registry import promote_best
from config import PREDICTIONS_COLLECTION
try:
    from ml.scheduler import start_scheduler
    SCHEDULER_AVAILABLE = True
except Exception:
    SCHEDULER_AVAILABLE = False

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
            "/train - Train & log models to MLflow",
            "/predict/temp - Predict temperature",
            "/predict/weather - Predict weather condition",
            "/scheduler/start - Start APScheduler (retrain every 5 minutes)",
            "/monitor/eval - Evaluate Production models and log metrics",
            "/registry/promote - Promote best run to Production",
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


@app.post("/train")
def train_models():
    try:
        success = retrain_from_mongo(MONGODB_PASSWORD)
        return {"status": "ok" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/predict/temp")
def predict_temperature(limit: int = 100):
    try:
        df = get_weather_data_from_mongodb("raw_weather_data", MONGODB_PASSWORD)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for prediction")
        df = df.sort_values('timestamp').tail(limit)
        preds = predict(df)
        out = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        # optional: persist predictions
        try:
            from mongodb_config import mongodb_config
            if mongodb_config.connect(MONGODB_PASSWORD):
                coll = mongodb_config.get_collection(PREDICTIONS_COLLECTION)
                records = out.assign(pred_type='regression', inserted_at=datetime.now()).to_dict('records')
                coll.insert_many(records)
        except Exception:
            pass
        return {
            "count": len(out),
            "columns": list(out.columns),
            "data": out[['city','timestamp','temperature','pred_temperature']].tail(10).to_dict(orient='records')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/monitor/eval")
def evaluate_models(limit: int = 500, run_etl: bool = False, persist: bool = False):
    try:
        # Optionally refresh latest data via ETL
        if run_etl:
            try:
                raw = fetch_weather_data()
                if raw:
                    df_etl = transform_weather_data(raw)
                    save_to_csv(df_etl)
                    save_to_mongodb(df_etl, MONGODB_PASSWORD)
            except Exception as e:
                # Don't block evaluation if ETL fails; continue with existing data
                print(f"ETL before eval failed: {e}")
        df = get_weather_data_from_mongodb("raw_weather_data", MONGODB_PASSWORD)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for evaluation")
        df = df.sort_values('timestamp').tail(limit)
        if persist:
            metrics, details = evaluate_with_details(df)
            # Persist predictions vs actuals batch for traceability
            try:
                from mongodb_config import mongodb_config
                if mongodb_config.connect(MONGODB_PASSWORD):
                    coll = mongodb_config.get_collection(PREDICTIONS_COLLECTION)
                    from datetime import datetime
                    records = details.assign(pred_type='eval', inserted_at=datetime.now()).to_dict('records')
                    if records:
                        coll.insert_many(records)
            except Exception:
                pass
            return {"status": "ok", "metrics": metrics, "persisted": True, "rows": int(len(details))}
        else:
            metrics = evaluate_and_log(df)
            return {"status": "ok", "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/registry/promote")
def registry_promote(task: str = "regression"):
    try:
        if task not in ("regression", "classification"):
            raise HTTPException(status_code=400, detail="task must be 'regression' or 'classification'")
        result = promote_best(task)  # promotes to Production
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")


@app.get("/predict/weather")
def predict_weather(limit: int = 100):
    try:
        df = get_weather_data_from_mongodb("raw_weather_data", MONGODB_PASSWORD)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for prediction")
        df = df.sort_values('timestamp').tail(limit)
        preds = predict(df)
        out = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        # optional: persist predictions
        try:
            from mongodb_config import mongodb_config
            if mongodb_config.connect(MONGODB_PASSWORD):
                coll = mongodb_config.get_collection(PREDICTIONS_COLLECTION)
                records = out.assign(pred_type='classification', inserted_at=datetime.now()).to_dict('records')
                coll.insert_many(records)
        except Exception:
            pass
        return {
            "count": len(out),
            "columns": list(out.columns),
            "data": out[['city','timestamp','weather','pred_condition']].tail(10).to_dict(orient='records')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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


@app.post("/scheduler/start")
def start_cron_scheduler():
    if not SCHEDULER_AVAILABLE:
        raise HTTPException(status_code=400, detail="Scheduler dependency not installed")
    try:
        start_scheduler()
        return {"status": "started", "job": "daily_retrain"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Optionally start scheduler when run as script
    if SCHEDULER_AVAILABLE and os.getenv("START_SCHEDULER", "0") == "1":
        start_scheduler()
    uvicorn.run(app, host="0.0.0.0", port=8000)