import jwt
import random
import string
import time
import logging
from datetime import datetime, timedelta
from config import JWT_SECRET, JWT_EXPIRY, SMS_2FA_TEST_NUMBER, EMAIL_2FA_TEST_ADDRESS

# Setup logging
logging.basicConfig(filename='logs/bot.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class LoginManager:
    def __init__(self):
        """Initialize LoginManager for JWT authentication and 2FA."""
        self.users = {}  # Simulated user database: {username: {'password': str, 'phone': str, 'email': str}}
        self.active_tokens = {}  # {token: {'username': str, 'expiry': timestamp}}

    def register_user(self, username, password, phone=None, email=None):
        """Register a new user (simulated, no real DB)."""
        if username in self.users:
            logger.error(f"Registration failed: Username {username} already exists")
            return False
        self.users[username] = {
            'password': password,
            'phone': phone or SMS_2FA_TEST_NUMBER,
            'email': email or EMAIL_2FA_TEST_ADDRESS
        }
        logger.info(f"User {username} registered successfully")
        return True

    def generate_2fa_code(self):
        """Generate a 6-digit 2FA code."""
        return ''.join(random.choices(string.digits, k=6))

    def send_2fa_code(self, username, method='sms'):
        """Simulate sending 2FA code via SMS or email."""
        if username not in self.users:
            logger.error(f"2FA failed: Username {username} not found")
            return None
        code = self.generate_2fa_code()
        if method == 'sms':
            logger.info(f"Sending SMS 2FA code {code} to {self.users[username]['phone']}")
            print(f"Test SMS 2FA code for {username}: {code} sent to {self.users[username]['phone']}")
        elif method == 'email':
            logger.info(f"Sending Email 2FA code {code} to {self.users[username]['email']}")
            print(f"Test Email 2FA code for {username}: {code} sent to {self.users[username]['email']}")
        return code

    def login(self, username, password, two_factor_code=None, two_factor_method='sms'):
        """Authenticate user and return JWT token if successful."""
        if username not in self.users or self.users[username]['password'] != password:
            logger.error(f"Login failed: Invalid username or password for {username}")
            return None

        # 2FA verification
        expected_code = self.send_2fa_code(username, method=two_factor_method)
        if two_factor_code != expected_code:
            logger.error(f"Login failed: Invalid 2FA code for {username}")
            return None

        # Generate JWT token
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(seconds=JWT_EXPIRY),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        self.active_tokens[token] = {'username': username, 'expiry': payload['exp'].timestamp()}
        logger.info(f"User {username} logged in successfully with token {token[:10]}...")
        return token

    def verify_token(self, token):
        """Verify JWT token and return username if valid."""
        if token not in self.active_tokens:
            logger.error("Token verification failed: Token not found")
            return None
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            if payload['exp'] < time.time():
                logger.error("Token verification failed: Token expired")
                del self.active_tokens[token]
                return None
            logger.info(f"Token verified for user {payload['username']}")
            return payload['username']
        except jwt.InvalidTokenError:
            logger.error("Token verification failed: Invalid token")
            del self.active_tokens[token]
            return None

if __name__ == "__main__":
    # Test the login system
    login_mgr = LoginManager()
    login_mgr.register_user("testuser", "password123", "1234567890", "test@example.com")
    token = login_mgr.login("testuser", "password123", "123456", two_factor_method='sms')  # Use printed code
    if token:
        print(f"JWT Token: {token}")
        user = login_mgr.verify_token(token)
        print(f"Verified User: {user}")