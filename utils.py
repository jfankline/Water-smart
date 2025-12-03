import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import requests

class WaterAnalyzer:
    """AI-powered water analysis utilities"""
    
    @staticmethod
    def detect_leak(usage_history, current_reading):
        """
        Detect water leak using Isolation Forest algorithm
        Args:
            usage_history: List of previous water usage readings
            current_reading: Current water usage reading
        Returns:
            tuple: (is_leak, anomaly_score, confidence)
        """
        if len(usage_history) < 7:
            return False, 0, 0
        
        # Prepare data
        X = np.array(usage_history).reshape(-1, 1)
        
        # Train Isolation Forest
        clf = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_estimators=100
        )
        clf.fit(X)
        
        # Predict anomaly
        current_array = np.array([current_reading]).reshape(-1, 1)
        prediction = clf.predict(current_array)
        anomaly_score = clf.decision_function(current_array)[0]
        
        # Convert score to confidence (0-100%)
        confidence = abs(anomaly_score) * 100
        
        is_leak = prediction[0] == -1
        return is_leak, anomaly_score, confidence
    
    @staticmethod
    def calculate_water_projection(current_water, daily_usage, days_to_project=90):
        """
        Calculate water projection for drought planning
        Args:
            current_water: Current water available in liters
            daily_usage: Average daily water usage in liters
            days_to_project: Number of days to project
        Returns:
            dict: Projection data
        """
        if daily_usage <= 0:
            return {
                'days_remaining': float('inf'),
                'projection': [],
                'critical_day': None
            }
        
        days_remaining = current_water / daily_usage
        
        # Create projection
        projection = []
        water_left = current_water
        
        for day in range(min(days_to_project, int(days_remaining) + 10)):
            water_left -= daily_usage
            if water_left < 0:
                water_left = 0
            
            projection.append({
                'day': day + 1,
                'projected_water': round(water_left, 2),
                'status': 'critical' if water_left < (daily_usage * 7) else 'low' if water_left < (daily_usage * 30) else 'adequate'
            })
        
        return {
            'days_remaining': round(days_remaining, 1),
            'projection': projection,
            'critical_day': int(days_remaining) if days_remaining > 0 else None
        }
    
    @staticmethod
    def calculate_water_balance(rainfall_mm, surface_area_m2, current_storage):
        """
        Calculate water balance after rainfall
        Args:
            rainfall_mm: Rainfall in millimeters
            surface_area_m2: Collection surface area in square meters
            current_storage: Current water storage in liters
        Returns:
            dict: Water balance data
        """
        # Convert rainfall mm to liters (1mm rain on 1m² = 1 liter)
        collected_water = rainfall_mm * surface_area_m2
        
        return {
            'rainfall_mm': rainfall_mm,
            'surface_area_m2': surface_area_m2,
            'collected_liters': collected_water,
            'current_storage': current_storage,
            'new_storage': current_storage + collected_water
        }

class WeatherService:
    """Weather data utilities"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, lat, lon):
        """Get current weather data"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'rainfall_1h': data.get('rain', {}).get('1h', 0),
                    'rainfall_3h': data.get('rain', {}).get('3h', 0),
                    'description': data['weather'][0]['description'],
                    'icon': data['weather'][0]['icon'],
                    'wind_speed': data['wind']['speed']
                }
        except Exception as e:
            print(f"Weather API error: {e}")
        
        return None
    
    def get_forecast(self, lat, lon, days=5):
        """Get weather forecast"""
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 forecasts per day
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                forecasts = []
                
                for item in data['list']:
                    forecasts.append({
                        'datetime': item['dt_txt'],
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'rainfall': item.get('rain', {}).get('3h', 0),
                        'description': item['weather'][0]['description']
                    })
                
                return forecasts
        except Exception as e:
            print(f"Weather forecast error: {e}")
        
        return None

class SMSNotifier:
    """SMS notification utilities"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def send_alert(self, phone_number, message):
        """Send SMS alert"""
        # This is a template for Africa's Talking API
        # In production, implement actual SMS gateway integration
        
        # For Africa's Talking:
        # import africastalking
        # africastalking.initialize(self.api_key, self.api_secret)
        # sms = africastalking.SMS
        # response = sms.send(message, [phone_number])
        
        # For Twilio:
        # from twilio.rest import Client
        # client = Client(self.api_key, self.api_secret)
        # message = client.messages.create(
        #     body=message,
        #     from_='+1234567890',
        #     to=phone_number
        # )
        
        # For demo, just log the message
        print(f"SMS Alert to {phone_number}: {message}")
        return True
    
    def send_whatsapp(self, phone_number, message):
        """Send WhatsApp message"""
        # Template for WhatsApp Business API
        print(f"WhatsApp to {phone_number}: {message}")
        return True