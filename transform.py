import pandas as pd
from datetime import datetime

def transform_weather_data(raw_data):
    transformed_data = []
    
    for item in raw_data:
        record = {
            "city": item["name"],
            "country": item["sys"]["country"],
            "temperature": item["main"]["temp"], #regression output
            "feels_like": item["main"]["feels_like"], #regression output
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "weather": item["weather"][0]["main"], #class 1
            "description": item["weather"][0]["description"], #class 2
            "wind_speed": item["wind"]["speed"],
            "timestamp": item["dt"]
        }
        transformed_data.append(record)
    
    return pd.DataFrame(transformed_data)