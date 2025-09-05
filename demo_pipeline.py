"""
Demo script for MongoDB ETL Pipeline with sample weather data
This script demonstrates the pipeline functionality without requiring API keys
"""

import pandas as pd
from datetime import datetime, timedelta
import random
from load import save_to_mongodb, get_weather_data_from_mongodb, list_collections
from config import MONGODB_PASSWORD

def generate_sample_weather_data():
    """Generate sample weather data for testing"""
    # Use a subset of cities for demo (first 10 cities)
    from config import CITIES
    demo_cities = CITIES[:10]  # Use first 10 cities for demo
    
    weather_conditions = ["Clear", "Clouds", "Rain", "Mist", "Thunderstorm"]
    
    data = []
    for city in demo_cities:
        # Generate realistic weather data
        base_temp = random.uniform(15, 40)  # Base temperature
        record = {
            "city": city,
            "country": "Various",
            "temperature": round(base_temp + random.uniform(-5, 5), 2),
            "feels_like": round(base_temp + random.uniform(-3, 7), 2),
            "humidity": random.randint(40, 90),
            "pressure": random.randint(1000, 1020),
            "weather": random.choice(weather_conditions),
            "description": f"{random.choice(['light', 'heavy', 'moderate'])} {random.choice(weather_conditions).lower()}",
            "wind_speed": round(random.uniform(1, 15), 2),
            "timestamp": int(datetime.now().timestamp())
        }
        data.append(record)
    
    return pd.DataFrame(data)

def demo_complete_pipeline():
    """Demonstrate the complete ETL pipeline with sample data"""
    print("🚀 Starting MongoDB ETL Pipeline Demo")
    print("=" * 50)
    
    try:
        # 1. Generate sample data (simulating Extract + Transform)
        print("1️⃣ Generating sample weather data...")
        df = generate_sample_weather_data()
        print(f"✅ Generated data for {len(df)} cities")
        print(f"📊 Cities: {df['city'].unique().tolist()}")
        print(f"🌡️ Temperature range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
        
        # 2. Load to MongoDB
        print("\n2️⃣ Loading to MongoDB Atlas...")
        success = save_to_mongodb(df, MONGODB_PASSWORD)
        if not success:
            print("❌ Failed to load data to MongoDB")
            return False
        
        # 3. List all collections
        print("\n3️⃣ Listing MongoDB collections...")
        collections = list_collections(MONGODB_PASSWORD)
        
        # 4. Demonstrate data retrieval from different collections
        print("\n4️⃣ Retrieving data from different collections...")
        
        # Test raw data collection
        print("\n📋 Raw Weather Data Collection:")
        raw_data = get_weather_data_from_mongodb("raw_weather_data", MONGODB_PASSWORD)
        if raw_data is not None and not raw_data.empty:
            print(f"✅ Retrieved {len(raw_data)} records")
            print("Sample record:")
            print(raw_data.iloc[0][['city', 'temperature', 'humidity', 'weather']].to_dict())
        
        # Test current weather collection
        print("\n📋 Current Weather Collection:")
        current_data = get_weather_data_from_mongodb("current_weather", MONGODB_PASSWORD)
        if current_data is not None and not current_data.empty:
            print(f"✅ Retrieved {len(current_data)} current weather records")
        
        # Test statistics collection
        print("\n📋 Weather Statistics Collection:")
        stats_data = get_weather_data_from_mongodb("weather_statistics", MONGODB_PASSWORD)
        if stats_data is not None and not stats_data.empty:
            print(f"✅ Retrieved {len(stats_data)} statistics records")
            latest_stats = stats_data.iloc[-1]
            print("Latest statistics:")
            print(f"  - Total records: {latest_stats['total_records']}")
            print(f"  - Cities count: {latest_stats['cities_count']}")
            print(f"  - Avg temperature: {latest_stats['avg_temperature']:.1f}°C")
            print(f"  - Temperature range: {latest_stats['min_temperature']:.1f}°C - {latest_stats['max_temperature']:.1f}°C")
        
        # Test batch-specific collections
        print("\n📋 Batch-specific Collections:")
        collections = list_collections(MONGODB_PASSWORD)
        batch_collections = [col for col in collections if col.startswith('weather_batch_')]
        
        if batch_collections:
            # Test the most recent batch collection
            latest_batch = sorted(batch_collections)[-1]
            batch_data = get_weather_data_from_mongodb(latest_batch, MONGODB_PASSWORD)
            if batch_data is not None and not batch_data.empty:
                print(f"  - {latest_batch}: {len(batch_data)} records")
                if 'batch_info' in batch_data.columns:
                    batch_info = batch_data.iloc[0]['batch_info']
                    print(f"    └── Batch ID: {batch_info.get('batch_id', 'N/A')}")
                    print(f"    └── Cities in batch: {len(batch_info.get('cities_in_batch', []))}")
                # Sample weather data from batch
                sample_record = batch_data.iloc[0]
                print(f"    └── Sample: {sample_record['city']} - {sample_record['temperature']}°C, {sample_record['weather']}")
        else:
            print("  - No batch collections found")
        
        print("\n🎉 Demo completed successfully!")
        print("\n📝 What was created:")
        print("✅ Multiple MongoDB collections with weather data")
        print("✅ Raw data storage for historical analysis")
        print("✅ Current weather data for real-time access")
        print("✅ Aggregated statistics for reporting")
        print("✅ Batch-specific collections for tracking ETL runs")
        print("✅ Enhanced statistics with temperature distribution")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_api_simulation():
    """Simulate FastAPI endpoints"""
    print("\n🔗 API Endpoints Simulation")
    print("=" * 30)
    
    print("Available endpoints:")
    print("📍 GET /run-etl-mongodb - Run complete ETL pipeline")
    print("📍 GET /weather-data/{collection_name} - Get data from collection")
    print("📍 GET /collections - List all collections")
    print("📍 GET /health - Health check")
    
    print("\nTo test the actual API:")
    print("1. Run: python main.py")
    print("2. Visit: http://localhost:8000/docs")
    print("3. Test endpoint: /run-etl-mongodb")

if __name__ == "__main__":
    # Run the demo
    success = demo_complete_pipeline()
    
    if success:
        demo_api_simulation()
        
        print("\n" + "=" * 50)
        print("🚀 Ready for Production!")
        print("📝 Next steps:")
        print("1. Add your OpenWeather API key to .env file")
        print("2. Run 'python main.py' to start FastAPI server")
        print("3. Test real weather data with /run-etl-mongodb endpoint")
    else:
        print("\n❌ Demo failed. Please check your MongoDB configuration.")
