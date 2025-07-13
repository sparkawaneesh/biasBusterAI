# Bias Buster - AI-Powered Bias Detection Tool

Bias Buster is a comprehensive web application that uses advanced AI to detect and analyze various types of bias in text content. It provides real-time analysis of written content, identifying gender, racial, political, and cultural biases with detailed reporting and actionable insights.

## Features

- **Multi-Input Analysis**: Analyze text directly or extract content from URLs
- **Comprehensive Bias Detection**: Identifies gender, racial, political, and cultural biases
- **AI-Powered Analysis**: Uses GROQ's LLaMA 3 model with keyword-based fallback
- **Detailed Reporting**: Provides specific issue descriptions, locations, and severity levels
- **Custom Model Training**: Complete ML pipeline for developing custom bias detection models
- **Modern UI**: Responsive design with smooth animations and gradient effects
- **Analysis History**: Track and review previous bias analyses

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for development and build tooling
- **Tailwind CSS** with shadcn/ui components
- **TanStack Query** for server state management
- **Wouter** for client-side routing

### Backend
- **Python 3.11** with Flask framework
- **GROQ API** for AI-powered bias detection
- **BeautifulSoup** for URL content extraction
- **Flask-CORS** for cross-origin requests

### Database
- **PostgreSQL** with Drizzle ORM
- **Neon Database** for serverless hosting
- **In-memory storage** fallback for development

## Getting Started

### Prerequisites
- Node.js 20 or higher
- Python 3.11 or higher
- GROQ API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

4. Start the Python backend:
   ```bash
   cd python_backend
   python3 app.py
   ```

5. In a new terminal, start the frontend:
   ```bash
   npm run dev
   ```

The application will be available at `http://localhost:5000`

## Usage

### Text Analysis
1. Navigate to the main page
2. Select the "Text" tab
3. Enter your text content
4. Choose analysis type and sensitivity level
5. Click "Analyze for Bias"

### URL Analysis
1. Navigate to the main page
2. Select the "URL" tab
3. Enter the URL of an article or blog post
4. Choose analysis type and sensitivity level
5. Click "Analyze for Bias"

### Understanding Results

The analysis provides:
- **Overall Bias Score**: General bias level (0-10 scale)
- **Category Scores**: Specific scores for gender, racial, political, and cultural bias
- **Detailed Issues**: Specific problems identified with descriptions and locations
- **Recommendations**: Actionable suggestions for improvement
- **Suggested Revisions**: Specific text improvements

## API Endpoints

### Health Check
```
GET /health
```

### Analyze Content
```
POST /api/analyze
Content-Type: application/json

{
  "content": "text to analyze",
  "analysisType": "comprehensive",
  "sensitivity": "standard",
  "inputType": "text"
}
```

### Get Analysis History
```
GET /api/analyses
```

### Get Specific Analysis
```
GET /api/analyses/:id
```

## Model Training

BiasGuard includes a complete ML training pipeline for developing custom bias detection models:

1. **Data Preparation**: Load and preprocess training datasets
2. **Model Architecture**: BERT-based multi-task bias detection
3. **Training Pipeline**: Comprehensive training with validation
4. **Evaluation**: Performance metrics and visualization
5. **Model Deployment**: Save and load trained models

See `ML_MODEL_TRAINING_GUIDE.md` for detailed instructions.

## Development

### Project Structure
```
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Application pages
│   │   ├── lib/            # Utility functions
│   │   └── hooks/          # Custom React hooks
├── python_backend/         # Python Flask backend
│   ├── app.py             # Main Flask application
│   ├── bias_detector.py   # Core bias detection logic
│   ├── model_training.py  # ML model training pipeline
│   └── url_scraper.py     # URL content extraction
├── server/                # Node.js server (legacy)
├── shared/                # Shared TypeScript schemas
└── components.json        # shadcn/ui configuration
```

### Available Scripts

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run start`: Start production server
- `npm run check`: Run TypeScript type checking
- `npm run db:push`: Push database schema changes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please refer to the documentation or create an issue in the repository.