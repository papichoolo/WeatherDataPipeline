import pandas as pd
from datetime import datetime
import os
from mongodb_config import mongodb_config
from pymongo.errors import BulkWriteError
import json

def save_to_csv(df, filename=None):
    """Keep the original CSV save functionality as backup"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_data_{timestamp}.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath

def save_to_mongodb(df, db_password: str):
    """Save weather data to MongoDB Atlas in different collections"""
    try:
        # Connect to MongoDB
        if not mongodb_config.connect(db_password):
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Add timestamp to all records
        df['inserted_at'] = datetime.now()
        df['batch_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Collection 1: Raw weather data (all records)
        raw_collection = mongodb_config.get_collection("raw_weather_data")
        if raw_collection is None:
            print("❌ Failed to get raw_weather_data collection")
            return False
        result_raw = raw_collection.insert_many(records)
        print(f"✅ Inserted {len(result_raw.inserted_ids)} records into 'raw_weather_data' collection")
        
        # Collection 2: Current weather (latest data per city)
        current_collection = mongodb_config.get_collection("current_weather")
        if current_collection is None:
            print("❌ Failed to get current_weather collection")
            return False
        
        # Delete existing current data and insert new
        current_collection.delete_many({})
        current_records = []
        
        for record in records:
            current_record = record.copy()
            current_record['is_current'] = True
            current_record['updated_at'] = datetime.now()
            current_records.append(current_record)
        
        result_current = current_collection.insert_many(current_records)
        print(f"✅ Inserted {len(result_current.inserted_ids)} records into 'current_weather' collection")
        
        # Collection 3: Batch-specific collection (organized by batch)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_collection_name = f"weather_batch_{batch_id}"
        batch_collection = mongodb_config.get_collection(batch_collection_name)
        if batch_collection is None:
            print(f"❌ Failed to get {batch_collection_name} collection")
        else:
            # Add batch metadata to records
            batch_records = []
            for record in records:
                batch_record = record.copy()
                batch_record['batch_info'] = {
                    'batch_id': batch_id,
                    'total_cities': len(df),
                    'cities_in_batch': df['city'].unique().tolist(),
                    'batch_timestamp': datetime.utcnow()
                }
                batch_records.append(batch_record)
            
            result_batch = batch_collection.insert_many(batch_records)
            print(f"✅ Inserted {len(result_batch.inserted_ids)} records into '{batch_collection_name}' collection")
        
        # Collection 4: Weather statistics (aggregated data)
        stats_collection = mongodb_config.get_collection("weather_statistics")
        if stats_collection is None:
            print("❌ Failed to get weather_statistics collection")
            return False
        
        # Calculate statistics
        stats = {
            'batch_id': batch_id,
            'timestamp': datetime.now(),
            'total_records': len(df),
            'cities_count': df['city'].nunique(),
            'avg_temperature': float(df['temperature'].mean()),
            'max_temperature': float(df['temperature'].max()),
            'min_temperature': float(df['temperature'].min()),
            'avg_humidity': float(df['humidity'].mean()),
            'cities': df['city'].unique().tolist(),
            'weather_conditions': df['weather'].value_counts().to_dict(),
            'batch_collection': batch_collection_name,
            'temperature_distribution': {
                'very_cold': len(df[df['temperature'] < 0]),
                'cold': len(df[(df['temperature'] >= 0) & (df['temperature'] < 10)]),
                'cool': len(df[(df['temperature'] >= 10) & (df['temperature'] < 20)]),
                'moderate': len(df[(df['temperature'] >= 20) & (df['temperature'] < 30)]),
                'warm': len(df[(df['temperature'] >= 30) & (df['temperature'] < 40)]),
                'hot': len(df[df['temperature'] >= 40])
            }
        }
        
        stats_collection.insert_one(stats)
        print(f"✅ Inserted statistics into 'weather_statistics' collection")
        
        return True
        
    except BulkWriteError as e:
        print(f"❌ Bulk write error: {e.details}")
        return False
    except Exception as e:
        print(f"❌ Error saving to MongoDB: {e}")
        return False
    finally:
        mongodb_config.close_connection()

def get_weather_data_from_mongodb(collection_name: str, db_password: str, query: dict = None):
    """Retrieve weather data from MongoDB"""
    try:
        if not mongodb_config.connect(db_password):
            return None
        
        collection = mongodb_config.get_collection(collection_name)
        
        if query is None:
            query = {}
        
        # Get data and convert to DataFrame
        cursor = collection.find(query)
        data = list(cursor)
        
        if data:
            # Remove MongoDB's _id field for cleaner DataFrame
            for record in data:
                record.pop('_id', None)
            
            df = pd.DataFrame(data)
            print(f"✅ Retrieved {len(df)} records from '{collection_name}' collection")
            return df
        else:
            print(f"❌ No data found in '{collection_name}' collection")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error retrieving data from MongoDB: {e}")
        return None
    finally:
        mongodb_config.close_connection()

def list_collections(db_password: str):
    """List all collections in the database"""
    try:
        if not mongodb_config.connect(db_password):
            return []
        
        collections = mongodb_config.database.list_collection_names()
        print(f"📋 Available collections: {collections}")
        return collections
        
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []
    finally:
        mongodb_config.close_connection()