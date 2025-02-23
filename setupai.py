import os
from cryptography.fernet import Fernet
import json
import logging
import sys

# Define project root directory
PROJECT_ROOT = "AiKrakBot"

# Define required directories
DIRECTORIES = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "backend"),
    os.path.join(PROJECT_ROOT, "backend", "ml_engine"),
    os.path.join(PROJECT_ROOT, "backend", "strategies"),
    os.path.join(PROJECT_ROOT, "frontend"),
    os.path.join(PROJECT_ROOT, "logs"),
    os.path.join(PROJECT_ROOT, "profiles")
]

# Define initial log files
LOG_FILES = [
    os.path.join(PROJECT_ROOT, "logs", "bot.log"),
    os.path.join(PROJECT_ROOT, "logs", "gan_metrics.log"),
    os.path.join(PROJECT_ROOT, "logs", "profile.log")
]

def setup_directories():
    """Create all required directories for the project structure."""
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/Verified directory: {directory}")

def setup_log_files():
    """Initialize empty log files if they don’t already exist."""
    for log_file in LOG_FILES:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("")
            print(f"Initialized log file: {log_file}")
        else:
            print(f"Log file already exists: {log_file}")

def generate_encryption_key():
    """Generate a secure Fernet encryption key."""
    return Fernet.generate_key()  # Returns bytes, 32 url-safe base64-encoded

def encrypt_credentials(api_key, api_secret, encryption_key):
    """Encrypt Kraken API key and secret using the Fernet encryption key."""
    cipher = Fernet(encryption_key)  # Expects bytes
    encrypted_api_key = cipher.encrypt(api_key.encode()).decode()
    encrypted_api_secret = cipher.encrypt(api_secret.encode()).decode()
    return encrypted_api_key, encrypted_api_secret

def create_env_file(api_key, api_secret, newsapi_key, simulate=True):
    """Create or update the .env file with encrypted credentials and other settings."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        overwrite = input(f".env file already exists at {env_path}. Overwrite? (yes/no): ").strip().lower()
        if overwrite != 'yes':
            print("Skipping .env creation. Using existing file.")
            return

    encryption_key = generate_encryption_key()  # Bytes
    encrypted_api_key, encrypted_api_secret = encrypt_credentials(api_key, api_secret, encryption_key)
    
    # Store encryption key as-is (base64-encoded bytes decoded to string)
    env_content = f"""# AiKrakBot Environment Variables
ENCRYPTED_KRAKEN_API_KEY={encrypted_api_key}
ENCRYPTED_KRAKEN_API_SECRET={encrypted_api_secret}
NEWSAPI_KEY={newsapi_key}
ENCRYPTION_KEY={encryption_key.decode('utf-8')}  # Decode bytes to string for .env
SIMULATE={simulate}
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    print(f"Created/Updated .env file at {env_path}")
    return encryption_key  # Return key for immediate use in state creation

def create_initial_state(encryption_key):
    """Create an initial encrypted state.json file using the provided encryption key."""
    state_path = os.path.join(PROJECT_ROOT, "state.json")
    if os.path.exists(state_path):
        overwrite = input(f"state.json already exists at {state_path}. Overwrite? (yes/no): ").strip().lower()
        if overwrite != 'yes':
            print("Skipping state.json creation. Using existing file.")
            return

    initial_state = {
        "trades": {"wins": 0, "losses": 0},
        "initial_balance": 10000.0  # Default starting balance in USD for simulation
    }
    
    # Use the provided encryption key directly (bytes)
    cipher = Fernet(encryption_key)  # Expects bytes
    encrypted_state = cipher.encrypt(json.dumps(initial_state).encode())
    
    with open(state_path, 'wb') as f:
        f.write(encrypted_state)
    print(f"Created initial state.json at {state_path}")

def validate_input(value, name):
    """Validate that a required input is provided."""
    if not value.strip():
        raise ValueError(f"{name} is required. Please provide a valid value.")

def main():
    """Main function to set up the AiKrakBot project environment."""
    print("=== AiKrakBot Project Setup ===")
    
    # Setup directories and log files
    setup_directories()
    setup_log_files()
    
    # Prompt for user input
    print("\nProvide the following information (press Enter to use defaults where applicable):")
    
    api_key = input("Kraken API Key (leave blank for simulation): ").strip()
    api_secret = input("Kraken API Secret (leave blank for simulation): ").strip()
    newsapi_key = input("NewsAPI Key (required, get from https://newsapi.org/): ").strip()
    simulate_input = input("Run in simulation mode? (yes/no, default yes): ").strip().lower() or "yes"
    simulate = simulate_input == "yes"
    
    # Validate NewsAPI key
    validate_input(newsapi_key, "NewsAPI Key")
    
    # Handle Kraken API credentials
    if simulate and not (api_key and api_secret):
        api_key = "dummy_api_key_for_simulation"
        api_secret = "dummy_api_secret_for_simulation"
        print("Simulation mode enabled. Using dummy Kraken API credentials.")
    elif not simulate:
        validate_input(api_key, "Kraken API Key")
        validate_input(api_secret, "Kraken API Secret")
    
    # Create .env file and get encryption key
    encryption_key = create_env_file(api_key, api_secret, newsapi_key, simulate)
    
    # Create initial state.json using the same encryption key
    create_initial_state(encryption_key)
    
    # Final instructions
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Ensure all project files are in AiKrakBot/ as per README.md.")
    print("2. Install dependencies:")
    print("   cd AiKrakBot")
    print("   pip install -r requirements.txt")
    print("3. Run the project:")
    print("   python main.py")
    print("Note: Check .env and state.json in AiKrakBot/ for setup details.")

if __name__ == "__main__":
    # Configure basic logging for setup errors
    logging.basicConfig(filename='setup_errors.log', level=logging.ERROR)
    try:
        main()
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        logging.error(f"Setup failed: {str(e)}")
        sys.exit(1)