import unittest
import json
from app import app, db
from models import User, WaterSource

class WaterSmartAPITestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        
        with app.app_context():
            db.create_all()
            
            # Create test user
            test_user = User(
                name='Test Farmer',
                phone='+254700000000',
                location='Test Location',
                farm_size=5.0
            )
            test_user.set_password('testpass')
            db.session.add(test_user)
            db.session.commit()
            
            self.test_user_id = test_user.id
    
    def tearDown(self):
        """Clean up after tests"""
        with app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_index(self):
        """Test API index endpoint"""
        response = self.app.get('/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', data)
    
    def test_register_user(self):
        """Test user registration"""
        user_data = {
            'name': 'New Farmer',
            'phone': '+254711111111',
            'location': 'Nairobi',
            'farm_size': 10.5,
            'password': 'newpass123'
        }
        
        response = self.app.post('/api/auth/register',
                                data=json.dumps(user_data),
                                content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 201)
        self.assertIn('access_token', data)
    
    def test_login_user(self):
        """Test user login"""
        login_data = {
            'phone': '+254700000000',
            'password': 'testpass'
        }
        
        response = self.app.post('/api/auth/login',
                                data=json.dumps(login_data),
                                content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', data)
    
    def test_add_water_source(self):
        """Test adding water source"""
        # First login to get token
        login_data = {'phone': '+254700000000', 'password': 'testpass'}
        login_response = self.app.post('/api/auth/login',
                                      data=json.dumps(login_data),
                                      content_type='application/json')
        token = json.loads(login_response.data)['access_token']
        
        # Add water source
        source_data = {
            'name': 'Test Tank',
            'capacity': 10000,
            'source_type': 'tank',
            'current_level': 5000
        }
        
        response = self.app.post('/api/water-sources',
                                data=json.dumps(source_data),
                                content_type='application/json',
                                headers={'Authorization': f'Bearer {token}'})
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 201)
        self.assertEqual(data['source']['name'], 'Test Tank')

if __name__ == '__main__':
    unittest.main()