import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(basedir, "..", "instance", "bank_statements.db")}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    UPLOAD_FOLDER = os.path.join(basedir, '..', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max
    ALLOWED_EXTENSIONS = {'pdf'}

    # Image extraction settings
    EXTRACTED_IMAGES_FOLDER = os.path.join(basedir, '..', 'extracted_images')
    PDF_DPI = 300  # Higher = better quality but larger files
    IMAGE_FORMAT = 'PNG'
    MAX_PAGES = 100  # Safety limit

    # GroqCloud settings
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.2-90b-vision-preview')
    GROQ_MAX_TOKENS = 8192  # Increased for long transaction lists
    GROQ_TIMEOUT = 120  # seconds - increased for complex pages

    # Processing settings
    CLEANUP_TEMP_FILES = os.environ.get('CLEANUP_TEMP_FILES', 'true').lower() == 'true'
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds

    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')

    # Executor
    EXECUTOR_TYPE = 'thread'
    EXECUTOR_MAX_WORKERS = 4


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ECHO = False


class ProductionConfig(Config):
    DEBUG = False
    # Use more secure settings


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
