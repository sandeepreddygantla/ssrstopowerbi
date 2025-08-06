#!/usr/bin/env python3
"""
RDL to Power BI Migration Tool - Application Launcher
Handles environment setup and graceful startup
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check and setup environment variables"""
    print("🔍 Checking environment configuration...")
    
    # Check for .env file
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("⚠️  .env file not found. Please copy .env.example to .env and configure your settings.")
        print(f"   Run: cp {env_example} {env_file}")
        return False
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("⚠️  OpenAI API key not configured.")
        print("   - AI features will be disabled")
        print("   - Set OPENAI_API_KEY in your .env file to enable AI-powered migration")
        print("   - Get your API key from: https://platform.openai.com/api-keys")
    else:
        print("✅ OpenAI API key configured - AI features enabled")
    
    # Check base URL path
    base_url = os.getenv('BASE_URL_PATH', '/rdlmigration')
    print(f"🌐 Base URL path: {base_url}")
    
    return True

def main():
    """Main application launcher"""
    print("🚀 RDL to Power BI Migration Tool")
    print("=" * 50)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not installed. Install requirements first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Import and run the application
    try:
        print("\n🏗️  Starting application...")
        from web_app import app, socketio, Config
        
        # Get configuration
        config_instance = Config()
        
        print(f"📂 Upload folder: {config_instance.UPLOAD_FOLDER}")
        print(f"📂 Results folder: {config_instance.RESULTS_FOLDER}")
        print(f"🌐 Server: {config_instance.HOST}:{config_instance.PORT}")
        print(f"🔗 Base URL: {config_instance.BASE_URL_PATH}")
        print(f"🐛 Debug mode: {config_instance.DEBUG}")
        
        print("\n" + "=" * 50)
        print(f"🌍 Application ready at: http://{config_instance.HOST}:{config_instance.PORT}{config_instance.BASE_URL_PATH}")
        print("📋 Dashboard: AI-powered analysis and migration")
        print("🚀 Migration: Bulk RDL file processing")
        print("📊 Results: View and download migration results")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the application
        socketio.run(
            app, 
            debug=config_instance.DEBUG, 
            host=config_instance.HOST, 
            port=config_instance.PORT, 
            use_reloader=config_instance.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env configuration")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Ensure Python 3.8+ is installed")
        sys.exit(1)

if __name__ == "__main__":
    main()