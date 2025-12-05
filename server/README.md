# Bank Statement Parser - Backend

Flask-based REST API for AI-powered bank statement extraction.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the server
python run.py
```

The API will be available at `http://localhost:5000`

## Fix Transaction Types (if needed)

If you have existing statements with incorrect transaction types:

```bash
source venv/bin/activate
python fix_transaction_types.py
```

## Full Documentation

For complete documentation including:
- Architecture details
- API endpoints
- Configuration options
- Troubleshooting
- Development guidelines

See the [main README](../README.md) in the project root.
