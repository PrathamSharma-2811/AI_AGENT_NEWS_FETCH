import pytest
import json
from app.main import app  # Import your Flask app

@pytest.fixture
def client():
    """A pytest fixture that provides a Flask test client."""
    app.config['TESTING'] = True  # Enable testing mode
    with app.test_client() as client:
        yield client

def test_login_success(client):
    """Test the login functionality with valid credentials."""
    response = client.post('/login', json={
        'username': 'testuser',
        'password': 'testpassword'
    })

    # Assert the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response contains a JWT token
    data = json.loads(response.data)
    assert 'token' in data

def test_login_failure(client):
    """Test the login functionality with invalid credentials."""
    response = client.post('/login', json={
        'username': 'wronguser',
        'password': 'wrongpassword'
    })

    # Assert the response status code is 401 (Unauthorized)
    assert response.status_code == 401



