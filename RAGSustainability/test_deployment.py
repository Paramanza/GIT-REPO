#!/usr/bin/env python3
"""
Simple test script to verify deployment configuration.
This helps debug Docker and Fly.io environment issues.
"""

import os
import requests
import time

def test_environment():
    """Test environment variable detection."""
    print("🔍 Environment Variable Test")
    print("=" * 40)
    
    env_vars = [
        'DOCKER_ENV',
        'FLY_APP_NAME', 
        'FLY_ALLOC_ID',
        'OPENAI_API_KEY',
        'PYTHONUNBUFFERED'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'NOT SET')
        if var == 'OPENAI_API_KEY' and value != 'NOT SET':
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"   {var} = {value}")
    
    print(f"\n🐳 Docker detection:")
    print(f"   /.dockerenv exists: {os.path.exists('/.dockerenv')}")
    print(f"   Running in container: {os.path.exists('/proc/1/cgroup')}")

def test_docker_detection():
    """Test the same Docker detection logic as app.py."""
    is_docker = (
        os.getenv('DOCKER_ENV') == 'true' or 
        os.path.exists('/.dockerenv') or 
        os.getenv('FLY_APP_NAME') is not None
    )
    
    print(f"\n🎯 Docker Detection Result: {is_docker}")
    
    if is_docker:
        print("✅ Should run in Docker mode (bind to 0.0.0.0:7860)")
    else:
        print("❌ Will run in local mode (this is wrong for deployment)")

def test_local_connection():
    """Test if the app is responding locally."""
    print(f"\n🌐 Connection Test")
    print("=" * 40)
    
    test_urls = [
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://0.0.0.0:7860"
    ]
    
    for url in test_urls:
        try:
            print(f"Testing {url}...")
            response = requests.get(url, timeout=5)
            print(f"   ✅ {url} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {url} - Error: {e}")

def main():
    """Main test function."""
    print("🧪 RAG Sustainability Deployment Test")
    print("=" * 50)
    
    test_environment()
    test_docker_detection()
    
    # Only test connections if we can import requests
    try:
        test_local_connection()
    except ImportError:
        print("\n⚠️  requests not available, skipping connection test")
    
    print("\n" + "=" * 50)
    print("🏁 Test complete!")

if __name__ == "__main__":
    main() 