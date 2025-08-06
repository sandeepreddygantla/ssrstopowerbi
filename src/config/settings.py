#!/usr/bin/env python3
"""
Configuration settings for RDL to Power BI Migration Tool
Centralized URL configuration for IIS deployment
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    def __init__(self):
        # Base URL Configuration - Change this single setting to update all URLs
        self.BASE_URL_PATH = os.getenv('BASE_URL_PATH', '/rdlmigration')
        
        # Flask configuration
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'rdl-migration-secret-key-2025')
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB max upload
        
        # Directory configuration
        self.BASE_DIR = Path(__file__).parent
        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
        self.RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', './results')
        
        # Server configuration
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', 5000))
        self.DEBUG = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
        
        # IIS deployment settings
        self.PREFERRED_URL_SCHEME = os.getenv('PREFERRED_URL_SCHEME', 'http')
        self.SEND_FILE_MAX_AGE_DEFAULT = 0  # Disable caching for development
        
        # CORS settings
        self.CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Derived URL paths (automatically use BASE_URL_PATH)
    def get_url(self, endpoint):
        """Generate full URL path for given endpoint"""
        if self.BASE_URL_PATH == '/':
            return f'/{endpoint}' if endpoint else '/'
        return f'{self.BASE_URL_PATH}/{endpoint}' if endpoint else self.BASE_URL_PATH
    
    def get_js_config(self):
        """Generate JavaScript configuration object"""
        return {
            'baseUrl': self.BASE_URL_PATH,
            'apiUrls': {
                'upload': self.get_url('api/upload'),
                'analyze': self.get_url('api/analyze'),
                'migrate': self.get_url('api/migrate'),
                'progress': self.get_url('api/progress'),
                'results': self.get_url('api/results'),
                'download': self.get_url('api/download'),
                'recentActivity': self.get_url('api/recent-activity'),
                'serviceStatus': self.get_url('api/service-status')
            },
            'pageUrls': {
                'dashboard': self.get_url(''),
                'migration': self.get_url('migration'),
                'results': self.get_url('results')
            }
        }

# Development configuration
class DevelopmentConfig(Config):
    def __init__(self):
        super().__init__()
        self.DEBUG = True
    
class ProductionConfig(Config):
    def __init__(self):
        super().__init__()
        self.DEBUG = False
        # Override for production IIS deployment
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', 80))

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'default')
    config_class = config.get(env, config['default'])
    return config_class()