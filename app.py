from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import datetime
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///watersmart.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Weather API Configuration
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'your-weather-api-key')
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# SMS Configuration
SMS_API_KEY = os.getenv('SMS_API_KEY', 'your-sms-api-key')
SMS_API_SECRET = os.getenv('SMS_API_SECRET', 'your-sms-api-secret')

# Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    location = db.Column(db.String(200))
    farm_size = db.Column(db.Float)
    county = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    water_sources = db.relationship('WaterSource', backref='owner', lazy=True)
    water_readings = db.relationship('WaterReading', backref='user', lazy=True)
    rainfall_data = db.relationship('RainfallData', backref='user', lazy=True)
    alerts = db.relationship('Alert', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class WaterSource(db.Model):
    __tablename__ = 'water_sources'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    source_type = db.Column(db.String(50))  # tank, pond, cistern, etc.
    capacity = db.Column(db.Float, nullable=False)  # in liters
    current_level = db.Column(db.Float, default=0)  # in liters
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    installed_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    readings = db.relationship('WaterReading', backref='source', lazy=True)

class WaterReading(db.Model):
    __tablename__ = 'water_readings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey('water_sources.id'), nullable=False)
    reading_value = db.Column(db.Float, nullable=False)  # in liters
    reading_type = db.Column(db.String(20))  # manual, meter, estimated
    reading_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    notes = db.Column(db.Text)
    is_leak_suspected = db.Column(db.Boolean, default=False)
    
    # Index for faster queries
    __table_args__ = (
        db.Index('idx_user_date', 'user_id', 'reading_date'),
    )

class RainfallData(db.Model):
    __tablename__ = 'rainfall_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)  # in mm
    date = db.Column(db.Date, nullable=False)
    source = db.Column(db.String(50))  # manual, weather_api
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # leak, low_water, rainfall
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20))  # low, medium, high
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    data = db.Column(db.JSON)  # Additional data in JSON format

class WaterUsagePattern(db.Model):
    __tablename__ = 'water_usage_patterns'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    daily_usage = db.Column(db.Float, nullable=False)
    predicted_usage = db.Column(db.Float)
    is_anomaly = db.Column(db.Boolean, default=False)
    anomaly_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ConservationTip(db.Model):
    __tablename__ = 'conservation_tips'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))  # irrigation, storage, general
    region = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Helper Functions
def detect_anomaly(user_id, current_reading, source_id=None):
    """Detect anomalies in water usage using Isolation Forest algorithm"""
    
    # Get last 30 days of readings
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    readings = WaterReading.query.filter(
        WaterReading.user_id == user_id,
        WaterReading.reading_date >= thirty_days_ago,
        WaterReading.reading_type == 'daily'
    ).order_by(WaterReading.reading_date).all()
    
    if len(readings) < 7:  # Need at least a week of data
        return False, 0
    
    # Prepare data for anomaly detection
    usage_data = [r.reading_value for r in readings]
    X = np.array(usage_data).reshape(-1, 1)
    
    # Train Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    
    # Predict anomaly for current reading
    current_array = np.array([current_reading]).reshape(-1, 1)
    prediction = clf.predict(current_array)
    anomaly_score = clf.decision_function(current_array)[0]
    
    is_anomaly = prediction[0] == -1
    return is_anomaly, anomaly_score

def calculate_water_projection(user_id):
    """Calculate how many days of water remain"""
    
    # Get all active water sources
    sources = WaterSource.query.filter_by(user_id=user_id, is_active=True).all()
    total_capacity = sum(source.capacity for source in sources)
    total_current = sum(source.current_level for source in sources)
    
    if total_current <= 0:
        return 0
    
    # Get average daily usage from last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_readings = WaterReading.query.filter(
        WaterReading.user_id == user_id,
        WaterReading.reading_date >= seven_days_ago,
        WaterReading.reading_type == 'daily'
    ).all()
    
    if recent_readings:
        avg_daily_usage = sum(r.reading_value for r in recent_readings) / len(recent_readings)
    else:
        avg_daily_usage = 100  # Default average usage in liters
    
    # Calculate days remaining
    days_remaining = total_current / avg_daily_usage if avg_daily_usage > 0 else 0
    return round(days_remaining, 1)

def get_weather_data(lat, lon):
    """Fetch weather data from OpenWeather API"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'rainfall': data.get('rain', {}).get('1h', 0),
                'description': data['weather'][0]['description']
            }
    except Exception as e:
        app.logger.error(f"Weather API error: {e}")
    
    return None

def send_sms_alert(phone, message):
    """Send SMS alert to farmer"""
    # This is a placeholder for actual SMS integration
    # In production, integrate with Twilio or Africa's Talking
    try:
        # Example with Africa's Talking (would need actual integration)
        # from africastalking.SMS import SMS
        # sms = SMS(username, api_key)
        # response = sms.send(message, [phone])
        
        app.logger.info(f"SMS sent to {phone}: {message}")
        return True
    except Exception as e:
        app.logger.error(f"SMS sending failed: {e}")
        return False

# Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'AI Water Leak & Rain-Harvest Manager API',
        'version': '1.0.0',
        'status': 'active'
    })

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'phone', 'location']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400
    
    # Check if user already exists
    if User.query.filter_by(phone=data['phone']).first():
        return jsonify({'error': 'Phone number already registered'}), 400
    
    # Create new user
    user = User(
        name=data['name'],
        phone=data['phone'],
        location=data['location'],
        farm_size=data.get('farm_size'),
        county=data.get('county'),
        email=data.get('email')
    )
    
    # Set password if provided
    if 'password' in data:
        user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'message': 'User registered successfully',
        'user': {
            'id': user.id,
            'name': user.name,
            'phone': user.phone,
            'location': user.location
        },
        'access_token': access_token
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate required fields
    if 'phone' not in data:
        return jsonify({'error': 'Phone number is required'}), 400
    
    # Find user
    user = User.query.filter_by(phone=data['phone']).first()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check password if provided
    if 'password' in data:
        if not user.check_password(data['password']):
            return jsonify({'error': 'Invalid password'}), 401
    else:
        # For SMS OTP login (simplified)
        # In production, generate and verify OTP here
        pass
    
    # Create access token
    access_token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user.id,
            'name': user.name,
            'phone': user.phone,
            'location': user.location
        },
        'access_token': access_token
    })

@app.route('/api/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'name': user.name,
        'phone': user.phone,
        'email': user.email,
        'location': user.location,
        'farm_size': user.farm_size,
        'county': user.county,
        'created_at': user.created_at.isoformat()
    })

# Water Sources Routes
@app.route('/api/water-sources', methods=['POST'])
@jwt_required()
def add_water_source():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'capacity']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400
    
    # Create new water source
    source = WaterSource(
        user_id=user_id,
        name=data['name'],
        source_type=data.get('source_type', 'tank'),
        capacity=float(data['capacity']),
        current_level=float(data.get('current_level', 0)),
        latitude=data.get('latitude'),
        longitude=data.get('longitude')
    )
    
    db.session.add(source)
    db.session.commit()
    
    return jsonify({
        'message': 'Water source added successfully',
        'source': {
            'id': source.id,
            'name': source.name,
            'type': source.source_type,
            'capacity': source.capacity,
            'current_level': source.current_level,
            'percentage': (source.current_level / source.capacity * 100) if source.capacity > 0 else 0
        }
    }), 201

@app.route('/api/water-sources', methods=['GET'])
@jwt_required()
def get_water_sources():
    user_id = get_jwt_identity()
    sources = WaterSource.query.filter_by(user_id=user_id, is_active=True).all()
    
    result = []
    for source in sources:
        result.append({
            'id': source.id,
            'name': source.name,
            'type': source.source_type,
            'capacity': source.capacity,
            'current_level': source.current_level,
            'percentage': (source.current_level / source.capacity * 100) if source.capacity > 0 else 0,
            'latitude': source.latitude,
            'longitude': source.longitude,
            'installed_date': source.installed_date.isoformat()
        })
    
    return jsonify({'sources': result})

# Water Readings Routes
@app.route('/api/water-readings', methods=['POST'])
@jwt_required()
def add_water_reading():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['source_id', 'reading_value']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400
    
    source_id = data['source_id']
    reading_value = float(data['reading_value'])
    
    # Verify source belongs to user
    source = WaterSource.query.filter_by(id=source_id, user_id=user_id).first()
    if not source:
        return jsonify({'error': 'Water source not found'}), 404
    
    # Detect anomaly
    is_anomaly, anomaly_score = detect_anomaly(user_id, reading_value, source_id)
    
    # Create reading
    reading = WaterReading(
        user_id=user_id,
        source_id=source_id,
        reading_value=reading_value,
        reading_type=data.get('reading_type', 'manual'),
        notes=data.get('notes'),
        is_leak_suspected=is_anomaly,
        reading_date=datetime.utcnow()
    )
    
    db.session.add(reading)
    
    # Update water source current level
    if data.get('update_current_level', True):
        source.current_level = reading_value
    
    # If anomaly detected, create alert
    if is_anomaly:
        alert = Alert(
            user_id=user_id,
            alert_type='leak',
            message=f'Abnormal water usage detected in {source.name}. Possible leak or theft.',
            severity='high',
            data={
                'source_id': source_id,
                'reading_value': reading_value,
                'anomaly_score': float(anomaly_score),
                'source_name': source.name
            }
        )
        db.session.add(alert)
        
        # Send SMS alert
        user = User.query.get(user_id)
        if user and user.phone:
            sms_message = f"ALERT: Possible leak detected in {source.name}. Water loss: {reading_value}L"
            send_sms_alert(user.phone, sms_message)
    
    db.session.commit()
    
    return jsonify({
        'message': 'Water reading recorded successfully',
        'reading': {
            'id': reading.id,
            'value': reading.reading_value,
            'date': reading.reading_date.isoformat(),
            'is_leak_suspected': reading.is_leak_suspected,
            'anomaly_score': float(anomaly_score) if anomaly_score else None
        }
    }), 201

@app.route('/api/water-readings/history', methods=['GET'])
@jwt_required()
def get_water_readings_history():
    user_id = get_jwt_identity()
    
    # Get query parameters
    days = request.args.get('days', default=7, type=int)
    source_id = request.args.get('source_id', type=int)
    
    # Calculate date range
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Build query
    query = WaterReading.query.filter(
        WaterReading.user_id == user_id,
        WaterReading.reading_date >= start_date
    )
    
    if source_id:
        query = query.filter_by(source_id=source_id)
    
    readings = query.order_by(WaterReading.reading_date.desc()).all()
    
    result = []
    for reading in readings:
        result.append({
            'id': reading.id,
            'source_id': reading.source_id,
            'source_name': reading.source.name if reading.source else 'Unknown',
            'value': reading.reading_value,
            'type': reading.reading_type,
            'date': reading.reading_date.isoformat(),
            'is_leak_suspected': reading.is_leak_suspected,
            'notes': reading.notes
        })
    
    return jsonify({'readings': result})

# Rainfall Routes
@app.route('/api/rainfall', methods=['POST'])
@jwt_required()
def add_rainfall_data():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    if 'amount' not in data:
        return jsonify({'error': 'Rainfall amount is required'}), 400
    
    # Create rainfall record
    rainfall = RainfallData(
        user_id=user_id,
        amount=float(data['amount']),
        date=datetime.utcnow().date(),
        source=data.get('source', 'manual'),
        latitude=data.get('latitude'),
        longitude=data.get('longitude')
    )
    
    db.session.add(rainfall)
    db.session.commit()
    
    return jsonify({
        'message': 'Rainfall data recorded successfully',
        'rainfall': {
            'id': rainfall.id,
            'amount': rainfall.amount,
            'date': rainfall.date.isoformat(),
            'source': rainfall.source
        }
    }), 201

@app.route('/api/weather/current', methods=['GET'])
@jwt_required()
def get_current_weather():
    user_id = get_jwt_identity()
    
    # Get user's location
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # For demo, use Nairobi coordinates if user doesn't have location
    lat = request.args.get('lat', -1.286389)
    lon = request.args.get('lon', 36.817223)
    
    weather_data = get_weather_data(lat, lon)
    
    if weather_data:
        # Store rainfall data if it's raining
        if weather_data['rainfall'] > 0:
            rainfall = RainfallData(
                user_id=user_id,
                amount=weather_data['rainfall'],
                date=datetime.utcnow().date(),
                source='weather_api',
                latitude=lat,
                longitude=lon
            )
            db.session.add(rainfall)
            db.session.commit()
        
        return jsonify({
            'weather': weather_data,
            'location': {'lat': lat, 'lon': lon}
        })
    else:
        return jsonify({'error': 'Could not fetch weather data'}), 500

# Dashboard Routes
@app.route('/api/dashboard/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    user_id = get_jwt_identity()
    
    # Get water sources
    sources = WaterSource.query.filter_by(user_id=user_id, is_active=True).all()
    total_capacity = sum(s.capacity for s in sources)
    total_current = sum(s.current_level for s in sources)
    
    # Get recent rainfall
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_rainfall = RainfallData.query.filter(
        RainfallData.user_id == user_id,
        RainfallData.date >= seven_days_ago.date()
    ).order_by(RainfallData.date.desc()).first()
    
    # Calculate water projection
    days_remaining = calculate_water_projection(user_id)
    
    # Get unread alerts
    unread_alerts = Alert.query.filter_by(
        user_id=user_id,
        is_read=False
    ).count()
    
    # Get recent anomalies
    recent_anomalies = WaterReading.query.filter(
        WaterReading.user_id == user_id,
        WaterReading.is_leak_suspected == True,
        WaterReading.reading_date >= (datetime.utcnow() - timedelta(days=7))
    ).count()
    
    return jsonify({
        'water_storage': {
            'total_capacity': total_capacity,
            'total_current': total_current,
            'percentage': (total_current / total_capacity * 100) if total_capacity > 0 else 0
        },
        'last_rainfall': recent_rainfall.amount if recent_rainfall else 0,
        'days_remaining': days_remaining,
        'unread_alerts': unread_alerts,
        'recent_leaks_detected': recent_anomalies
    })

@app.route('/api/dashboard/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    user_id = get_jwt_identity()
    
    alerts = Alert.query.filter_by(user_id=user_id)\
        .order_by(Alert.created_at.desc())\
        .limit(20)\
        .all()
    
    result = []
    for alert in alerts:
        result.append({
            'id': alert.id,
            'type': alert.alert_type,
            'message': alert.message,
            'severity': alert.severity,
            'is_read': alert.is_read,
            'created_at': alert.created_at.isoformat(),
            'data': alert.data
        })
    
    return jsonify({'alerts': result})

@app.route('/api/alerts/<int:alert_id>/read', methods=['PUT'])
@jwt_required()
def mark_alert_as_read(alert_id):
    user_id = get_jwt_identity()
    
    alert = Alert.query.filter_by(id=alert_id, user_id=user_id).first()
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404
    
    alert.is_read = True
    db.session.commit()
    
    return jsonify({'message': 'Alert marked as read'})

# Conservation Tips Routes
@app.route('/api/conservation-tips', methods=['GET'])
def get_conservation_tips():
    region = request.args.get('region', 'general')
    
    tips = ConservationTip.query.filter(
        (ConservationTip.region == region) | (ConservationTip.region == 'general')
    ).order_by(db.func.random()).limit(5).all()
    
    result = []
    for tip in tips:
        result.append({
            'id': tip.id,
            'title': tip.title,
            'description': tip.description,
            'category': tip.category
        })
    
    return jsonify({'tips': result})

# Drought Planning Routes
@app.route('/api/drought/plan', methods=['POST'])
@jwt_required()
def create_drought_plan():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    if 'expected_duration' not in data:
        return jsonify({'error': 'Expected drought duration is required'}), 400
    
    expected_duration = int(data['expected_duration'])  # in days
    
    # Get current water storage
    sources = WaterSource.query.filter_by(user_id=user_id, is_active=True).all()
    total_current = sum(s.current_level for s in sources)
    
    # Calculate daily allowance
    if expected_duration > 0 and total_current > 0:
        daily_allowance = total_current / expected_duration
    else:
        daily_allowance = 0
    
    # Get conservation tips
    user = User.query.get(user_id)
    region = user.county if user and user.county else 'general'
    tips = ConservationTip.query.filter(
        (ConservationTip.region == region) | (ConservationTip.region == 'general'),
        ConservationTip.category == 'drought'
    ).limit(3).all()
    
    tips_list = [{
        'title': tip.title,
        'description': tip.description
    } for tip in tips]
    
    return jsonify({
        'drought_plan': {
            'expected_duration_days': expected_duration,
            'current_water_liters': total_current,
            'recommended_daily_allowance_liters': round(daily_allowance, 2),
            'tips': tips_list
        }
    })

# Admin Routes (for NGOs/Government)
@app.route('/api/admin/aggregated-data', methods=['GET'])
def get_aggregated_data():
    # This endpoint would require admin authentication in production
    # For demo, we'll return sample aggregated data
    
    # Get total registered farmers
    total_farmers = User.query.count()
    
    # Get total water saved (estimated)
    total_leaks = WaterReading.query.filter_by(is_leak_suspected=True).count()
    estimated_water_saved = total_leaks * 1000  # Assume 1000L saved per leak detected
    
    # Get regional data
    regional_data = db.session.query(
        User.county,
        db.func.count(User.id).label('farmer_count'),
        db.func.sum(WaterSource.capacity).label('total_capacity')
    ).join(WaterSource, User.id == WaterSource.user_id)\
     .group_by(User.county).all()
    
    regions = []
    for region in regional_data:
        regions.append({
            'county': region.county or 'Unknown',
            'farmer_count': region.farmer_count,
            'total_capacity': float(region.total_capacity) if region.total_capacity else 0
        })
    
    return jsonify({
        'summary': {
            'total_farmers': total_farmers,
            'total_leaks_detected': total_leaks,
            'estimated_water_saved_liters': estimated_water_saved,
            'active_this_month': User.query.filter(
                User.created_at >= (datetime.utcnow() - timedelta(days=30))
            ).count()
        },
        'regional_data': regions
    })

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database and seed data
def init_db():
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Seed conservation tips if table is empty
        if ConservationTip.query.count() == 0:
            tips = [
                ConservationTip(
                    title='Use Drip Irrigation',
                    description='Drip irrigation delivers water directly to plant roots, reducing evaporation and runoff by up to 60%.',
                    category='irrigation',
                    region='general'
                ),
                ConservationTip(
                    title='Apply Mulch',
                    description='Mulching helps retain soil moisture, reduces evaporation, and suppresses weeds that compete for water.',
                    category='general',
                    region='general'
                ),
                ConservationTip(
                    title='Harvest Rainwater',
                    description='Collect rainwater from rooftops during rainy seasons to use during dry periods.',
                    category='storage',
                    region='general'
                ),
                ConservationTip(
                    title='Water Early Morning',
                    description='Water plants early in the morning to reduce evaporation losses from sun and wind.',
                    category='irrigation',
                    region='general'
                ),
                ConservationTip(
                    title='Fix Leaks Immediately',
                    description='A small leak can waste hundreds of liters per day. Check pipes and tanks regularly.',
                    category='storage',
                    region='general'
                ),
                ConservationTip(
                    title='Choose Drought-Resistant Crops',
                    description='Plant crops that require less water, such as millet, sorghum, or certain bean varieties.',
                    category='general',
                    region='arid'
                ),
                ConservationTip(
                    title='Use Shade Nets',
                    description='Shade nets reduce water evaporation from soil and protect plants from excessive sun.',
                    category='general',
                    region='hot'
                ),
                ConservationTip(
                    title='Group Plants by Water Needs',
                    description='Plant crops with similar water requirements together to optimize irrigation.',
                    category='irrigation',
                    region='general'
                ),
                ConservationTip(
                    title='Monitor Soil Moisture',
                    description='Check soil moisture before watering to avoid over-irrigation.',
                    category='irrigation',
                    region='general'
                ),
                ConservationTip(
                    title='Reuse Household Water',
                    description='Use water from washing vegetables or rinsing for watering plants.',
                    category='general',
                    region='general'
                )
            ]
            
            for tip in tips:
                db.session.add(tip)
            
            db.session.commit()
            print("Database initialized with sample data")

if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)