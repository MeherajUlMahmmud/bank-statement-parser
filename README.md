# Bank Statement Parser

An intelligent bank statement parsing application that uses AI-powered vision models to extract transaction data from PDF bank statements with high accuracy. The system handles various bank statement formats and provides structured data extraction with confidence scoring.

## Features

- **AI-Powered Extraction**: Leverages GroqCloud's vision models for intelligent data extraction
- **Multi-Format Support**: Handles different bank statement layouts and column structures
- **PDF Processing**: Automatically converts multi-page PDFs to images for analysis
- **Async Processing**: Parallel processing of multiple pages for improved performance
- **Confidence Scoring**: AI provides confidence scores for extracted data quality assessment
- **Data Validation**: Pydantic-based schema validation ensures data integrity
- **Token Tracking**: Monitors API usage and costs with detailed token statistics
- **Modern UI**: React-based frontend with Tailwind CSS for a clean user experience
- **RESTful API**: Well-structured Flask backend with comprehensive endpoints

## Architecture

```
bank-statement-parser/
├── backend/               # Flask REST API
│   ├── app/
│   │   ├── models.py     # Database models
│   │   ├── config.py     # Configuration management
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic
│   │   │   ├── llm_service.py     # AI extraction service
│   │   │   ├── pdf_processor.py   # PDF to image conversion
│   │   │   └── processor.py       # Statement processing
│   │   └── utils/        # Helper utilities
│   └── run.py           # Application entry point
│
├── frontend/             # React + Vite application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API client
│   │   └── types/       # TypeScript types
│   └── public/          # Static assets
│
├── start-backend.sh     # Backend startup script
└── start-frontend.sh    # Frontend startup script
```

## Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher
- **npm**: 9.x or higher
- **GroqCloud API Key**: Required for AI-powered extraction

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bank-statement-parser
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration:
#   - SECRET_KEY: Generate a secure random string
#   - GROQ_API_KEY: Your GroqCloud API key
#   - Other settings as needed

# Initialize database (if using migrations)
flask db upgrade
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
# Edit .env with your backend URL if different from default
```

## Configuration

### Backend Configuration (`.env`)

```env
# Flask Configuration
SECRET_KEY=your-secure-secret-key-here
FLASK_ENV=development
FLASK_DEBUG=1

# GroqCloud API
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct

# Database
DATABASE_URL=sqlite:///bank_statements.db

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Processing
CLEANUP_TEMP_FILES=true
```

### Frontend Configuration (`.env`)

```env
VITE_API_BASE_URL=http://localhost:5000
```

## Running the Application

### Quick Start (Using Scripts)

```bash
# From project root directory

# Start backend (Terminal 1)
./start-backend.sh

# Start frontend (Terminal 2)
./start-frontend.sh
```

### Manual Start

#### Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run.py
```

The backend will run on `http://localhost:5000`

#### Frontend

```bash
cd frontend
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. **Open the application** in your browser at `http://localhost:5173`
2. **Upload a bank statement PDF** using the upload interface
3. **Wait for processing** - the system will:
   - Convert PDF pages to images
   - Extract data using AI vision model
   - Validate and structure the data
   - Store results in the database
4. **View extracted data** including:
   - Customer and bank details
   - All transactions with original column names
   - Confidence scores for data quality
   - Token usage statistics

## API Endpoints

### Statements

- `POST /api/upload` - Upload a bank statement PDF
- `GET /api/statements` - List all processed statements
- `GET /api/statements/:id` - Get specific statement details
- `DELETE /api/statements/:id` - Delete a statement

### Transactions

- `GET /api/transactions` - List all transactions
- `GET /api/transactions?statement_id=:id` - Get transactions for a specific statement

## Key Features Explained

### AI-Powered Extraction

The system uses GroqCloud's vision models with:
- **System prompts** defining the AI's role as a financial document expert
- **Few-shot examples** showing the AI correct extraction patterns
- **Confidence scoring** for quality assessment
- **Structured output** with Pydantic validation

### Async Processing

Multiple PDF pages are processed in parallel using Python's `asyncio`:
- Significant performance improvement for multi-page statements
- Non-blocking I/O operations
- Efficient resource utilization

### Data Validation

Pydantic models ensure:
- Type safety for all extracted fields
- Required fields are present
- Data structure consistency
- Graceful handling of validation errors

### Token Tracking

Monitor API costs with:
- Per-page token usage
- Cumulative statistics
- Prompt and completion token breakdown
- Cost estimation capabilities

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate

# Run with auto-reload
FLASK_DEBUG=1 python run.py

# Run tests (if available)
pytest
```

### Frontend Development

```bash
cd frontend

# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Linting
npm run lint
```

## Project Structure Details

### Backend Services

- **llm_service.py**: AI extraction with confidence scoring and validation
- **pdf_processor.py**: PDF to image conversion using PyMuPDF
- **processor.py**: Orchestrates the extraction pipeline
- **parser.py**: Data parsing and transformation utilities

### Frontend Components

- **Upload**: PDF upload interface with drag-and-drop
- **StatementList**: Display processed statements
- **TransactionTable**: Interactive transaction data viewer
- **ConfidenceIndicator**: Visual confidence score display

## Technology Stack

### Backend

- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation
- **httpx**: Async HTTP client
- **PyMuPDF**: PDF processing
- **Pillow**: Image processing

### Frontend

- **React 19**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Axios**: HTTP client
- **React Router**: Navigation
- **Lucide React**: Icons

## Troubleshooting

### Backend Issues

**Issue**: `ModuleNotFoundError` for dependencies
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Issue**: Database errors
```bash
# Reset database
rm bank_statements.db
flask db upgrade
```

**Issue**: GroqCloud API errors
- Verify `GROQ_API_KEY` is correct in `.env`
- Check API quota limits
- Review logs for specific error messages

### Frontend Issues

**Issue**: Cannot connect to backend
- Verify backend is running on `http://localhost:5000`
- Check `VITE_API_BASE_URL` in frontend `.env`
- Verify CORS settings in backend `.env`

**Issue**: Build errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Performance Optimization

### For Large Statements

- **Async Processing**: Automatically enabled for multi-page PDFs
- **Image Optimization**: Consider compressing images before processing
- **Batch Processing**: Process multiple statements concurrently

### Token Usage Optimization

- Monitor token usage via logs and API response metadata
- Adjust `GROQ_MAX_TOKENS` if responses are truncated
- Use token statistics to estimate costs

## Security Considerations

- Store API keys in `.env` files (never commit to version control)
- Use strong `SECRET_KEY` for Flask sessions
- Implement authentication for production deployments
- Validate and sanitize all file uploads
- Set appropriate CORS origins for production

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Support for additional file formats (images, CSV)
- [ ] Export to Excel, CSV, or JSON
- [ ] Transaction categorization and analytics
- [ ] Multi-language statement support
- [ ] Real-time processing status updates via WebSockets
- [ ] Batch upload and processing
- [ ] Advanced search and filtering

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v2.0. See the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## Acknowledgments

- GroqCloud for providing the vision AI models
- The open-source community for the excellent tools and libraries
