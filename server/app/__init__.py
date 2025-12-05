import os
import logging
from logging.handlers import TimedRotatingFileHandler
from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from app.config import config
from app.models import db

# Initialize extensions
migrate = Migrate()


def setup_logging(app):
    """
    Configure logging with daily rotation and 30-day retention

    Creates rotating log files in the 'logs' directory with:
    - Daily rotation (new file at midnight)
    - 30-day retention (automatically deletes old logs)
    - Separate files for app logs and error logs
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Application log handler (INFO level and above) - daily rotation, 30 days retention
    app_log_file = os.path.join(log_dir, 'app.log')
    app_handler = TimedRotatingFileHandler(
        app_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    app_handler.suffix = '%Y-%m-%d'

    # Error log handler (ERROR level and above) - daily rotation, 30 days retention
    error_log_file = os.path.join(log_dir, 'error.log')
    error_handler = TimedRotatingFileHandler(
        error_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.suffix = '%Y-%m-%d'

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Configure app logger
    app.logger.addHandler(app_handler)
    app.logger.addHandler(error_handler)
    app.logger.setLevel(logging.INFO)

    app.logger.info(f"Logging configured - logs directory: {log_dir}")
    app.logger.info("Log rotation: daily at midnight, retention: 30 days")


def create_app(config_name='default'):
    """
    Flask application factory

    Args:
        config_name: Configuration to use ('development', 'production', 'default')

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Setup logging with daily rotation and 30-day retention
    setup_logging(app)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}})

    # Register blueprints
    from app.routes.upload import upload_bp
    from app.routes.statements import statements_bp
    from app.routes.transactions import transactions_bp

    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(statements_bp, url_prefix='/api')
    app.register_blueprint(transactions_bp, url_prefix='/api')

    # Create database tables
    with app.app_context():
        db.create_all()

    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return {'status': 'healthy', 'message': 'Bank Statement Parser API'}, 200

    # Root endpoint
    @app.route('/', methods=['GET'])
    def index():
        return {
            'name': 'Bank Statement Parser API',
            'version': '1.0.0',
            'endpoints': {
                'upload': '/api/upload',
                'statements': '/api/statements',
                'transactions': '/api/transactions'
            }
        }, 200

    return app
