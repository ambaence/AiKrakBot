from cryptography.fernet import Fernet
import base64
import os
import time
import logging
import random
import string
from config import ENCRYPTION_KEY, KEY_ROTATION_INTERVAL

# Setup logging
logging.basicConfig(filename='logs/bot.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        """Initialize SecurityManager for API key encryption and rotation."""
        self.cipher = Fernet(ENCRYPTION_KEY if ENCRYPTION_KEY else self.generate_encryption_key())
        self.api_key = None
        self.api_secret = None
        self.last_rotation = time.time()

    def generate_encryption_key(self):
        """Generate a new Fernet encryption key if not provided."""
        key = Fernet.generate_key()
        logger.info("Generated new encryption key")
        return key

    def encrypt_key(self, key):
        """Encrypt an API key or secret."""
        if isinstance(key, str):
            key = key.encode()
        encrypted_key = self.cipher.encrypt(key)
        logger.info("API key encrypted")
        return encrypted_key.decode()

    def decrypt_key(self, encrypted_key):
        """Decrypt an API key or secret."""
        if isinstance(encrypted_key, str):
            encrypted_key = encrypted_key.encode()
        decrypted_key = self.cipher.decrypt(encrypted_key)
        logger.info("API key decrypted")
        return decrypted_key.decode()

    def load_encrypted_keys(self, encrypted_api_key, encrypted_api_secret):
        """Load and decrypt API keys from config."""
        self.api_key = self.decrypt_key(encrypted_api_key)
        self.api_secret = self.decrypt_key(encrypted_api_secret)

    def generate_new_api_key_pair(self):
        """Simulate generating a new API key pair (replace with real Kraken API call in production)."""
        new_key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        new_secret = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        encrypted_key = self.encrypt_key(new_key)
        encrypted_secret = self.encrypt_key(new_secret)
        return encrypted_key, encrypted_secret

    def rotate_keys(self):
        """Rotate API keys if beyond rotation interval."""
        current_time = time.time()
        if current_time - self.last_rotation >= KEY_ROTATION_INTERVAL:
            logger.info("Rotating API keys...")
            new_encrypted_key, new_encrypted_secret = self.generate_new_api_key_pair()
            self.load_encrypted_keys(new_encrypted_key, new_encrypted_secret)
            self.last_rotation = current_time
            logger.info(f"API keys rotated. New encrypted key: {new_encrypted_key[:10]}..., secret: {new_encrypted_secret[:10]}...")
            return new_encrypted_key, new_encrypted_secret
        return None, None

    def get_api_credentials(self):
        """Return current decrypted API credentials, rotating if needed."""
        self.rotate_keys()
        return self.api_key, self.api_secret

if __name__ == "__main__":
    sec_mgr = SecurityManager()
    test_key = "test_api_key"
    test_secret = "test_api_secret"
    enc_key = sec_mgr.encrypt_key(test_key)
    enc_secret = sec_mgr.encrypt_key(test_secret)
    sec_mgr.load_encrypted_keys(enc_key, enc_secret)
    key, secret = sec_mgr.get_api_credentials()
    print(f"Decrypted Key: {key}, Secret: {secret}")
    time.sleep(2)
    new_key, new_secret = sec_mgr.rotate_keys()
    print(f"New Encrypted Key: {new_key}, Secret: {new_secret}")