import requests
from config import API_KEY, BASE_URL, CITIES

def fetch_weather_data():
    all_data = []
    
    for city in CITIES:
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"  # for Celsius
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            all_data.append(data)
        except requests.RequestException as e:
            print(f"Error fetching data for {city}: {e}")
    
    return all_data