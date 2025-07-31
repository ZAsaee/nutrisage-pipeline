# NutriSage MLOps Pipeline

A comprehensive machine learning pipeline for nutrition grade prediction using the Open Food Facts dataset.

## 🎯 Project Overview

NutriSage is an ML-powered nutrition grade prediction system that classifies food products into nutrition grades (A, B, C, D, E) based on their nutritional composition. This project demonstrates a complete MLOps pipeline from data preprocessing to model deployment.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Preprocessing  │    │   Training      │    │   Prediction    │
│   (Parquet)     │───▶│   Pipeline      │───▶│   XGBoost       │───▶│   API/CLI       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3/Local      │    │   Feature       │    │   Hyperparameter│    │   REST API      │
│   Storage       │    │   Engineering   │    │   Tuning        │    │   Endpoints     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- AWS CLI (for S3 data access)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd nutrisage-mlops
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp env.example .env
# Edit .env with your configuration
```

### Basic Usage

1. **Data Preprocessing**
```bash
python -m src.preprocessing --input data/raw/sample_data.parquet
```

2. **Model Training**
```bash
# Basic training
python -m src.modeling.train

# Training with hyperparameter tuning
python -m src.modeling.train --tune

# Training with sampling (faster)
python -m src.modeling.train --sample-fraction 0.1 --tune
```

3. **Model Prediction**
```bash
python -m src.modeling.predict --input data/raw/sample_data.parquet
```

4. **Generate Plots**
```bash
python -m src.plots --type all
```

## 📊 Data Pipeline

### Data Sources
- **Primary**: Open Food Facts database (S3: `s3://nutrisage/data/`)
- **Local**: Sample data in `data/raw/sample_data.parquet`

### Features
The model uses 10 nutritional features:
- `energy-kcal_100g` - Energy content per 100g
- `fat_100g` - Fat content per 100g
- `carbohydrates_100g` - Carbohydrate content per 100g
- `sugars_100g` - Sugar content per 100g
- `proteins_100g` - Protein content per 100g
- `sodium_100g` - Sodium content per 100g
- `fat_carb_ratio` - Fat to carbohydrate ratio
- `protein_carb_ratio` - Protein to carbohydrate ratio
- `protein_fat_ratio` - Protein to fat ratio
- `total_macros` - Sum of macronutrients

### Preprocessing Steps
1. **Data Loading**: Load from S3 or local storage
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create derived features and ratios
4. **Data Validation**: Ensure data quality and ranges
5. **Label Encoding**: Convert nutrition grades to numeric labels

## 🤖 Model Architecture

### Algorithm
- **XGBoost Classifier**: Gradient boosting for multi-class classification
- **Classes**: 5 nutrition grades (A, B, C, D, E)
- **Objective**: Multi-class softmax probability

### Hyperparameter Tuning
- **Method**: Grid Search with 3-fold Cross-Validation
- **Parameters**: max_depth, learning_rate, n_estimators, subsample, colsample_bytree
- **Optimization**: Accuracy maximization

### Model Performance
- **Test Accuracy**: [To be filled]
- **Cross-validation**: [To be filled]
- **Feature Importance**: Sodium and sugars are most predictive

## 🛠️ Development

### Project Structure
```
nutrisage-mlops/
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── modeling/          # Model training and prediction
│   │   ├── train.py       # Model training
│   │   └── predict.py     # Model prediction
│   ├── plots.py           # Visualization utilities
│   ├── api/               # REST API endpoints
│   └── config.py          # Configuration management
├── data/                  # Data files
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                # Trained models
├── reports/               # Reports and visualizations
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── Makefile              # Automation commands
└── README.md             # This file
```

### Key Commands

#### Data Pipeline
```bash
# Load data from S3
python -m src.dataset --source s3 --input s3://nutrisage/data/ --sample 0.1

# Load local data
python -m src.dataset --source local --input data/raw/sample_data.parquet

# Preprocess data
python -m src.preprocessing --input data/raw/sample_data.parquet
```

#### Model Training
```bash
# Basic training
python -m src.modeling.train

# Training with hyperparameter tuning
python -m src.modeling.train --tune

# Training with sampling
python -m src.modeling.train --sample-fraction 0.1 --tune
```

#### Model Prediction
```bash
# Batch prediction
python -m src.modeling.predict --input data/raw/sample_data.parquet

# Single prediction (via API)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"energy_kcal_100g": 150, "fat_100g": 5.2, ...}'
```

#### Visualization
```bash
# Generate all plots
python -m src.plots --type all

# Generate specific plots
python -m src.plots --type feature_importance
python -m src.plots --type confusion_matrix
```

### Makefile Commands
```bash
# Install dependencies
make install

# Run preprocessing
make preprocess

# Train model
make train

# Train with hyperparameter tuning
make train-tune

# Run predictions
make predict

# Generate plots
make plots

# Run tests
make test

# Clean up
make clean
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t nutrisage-mlops .
```

### Run Container
```bash
# Run with local data
docker run -p 8000:8000 -v $(pwd)/data:/app/data nutrisage-mlops

# Run with S3 access
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  nutrisage-mlops
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📈 API Documentation

### Endpoints

#### Health Check
```http
GET /api/health
```

#### Single Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "energy_kcal_100g": 150,
  "fat_100g": 5.2,
  "carbohydrates_100g": 25.0,
  "sugars_100g": 12.0,
  "proteins_100g": 8.0,
  "sodium_100g": 0.3
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "data": [
    {"energy_kcal_100g": 150, "fat_100g": 5.2, ...},
    {"energy_kcal_100g": 200, "fat_100g": 8.1, ...}
  ]
}
```

#### Model Info
```http
GET /model/info
```

### API Usage Examples

#### Python
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "energy_kcal_100g": 150,
    "fat_100g": 5.2,
    "carbohydrates_100g": 25.0,
    "sugars_100g": 12.0,
    "proteins_100g": 8.0,
    "sodium_100g": 0.3
})

print(response.json())
# {"prediction": "B", "confidence": 0.85, "probabilities": {...}}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "energy_kcal_100g": 150,
       "fat_100g": 5.2,
       "carbohydrates_100g": 25.0,
       "sugars_100g": 12.0,
       "proteins_100g": 8.0,
       "sodium_100g": 0.3
     }'
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_training_pipeline.py

# Run with coverage
pytest --cov=src tests/
```

### Test Structure
- `tests/test_data_pipeline.py` - Data loading and preprocessing tests
- `tests/test_training_pipeline.py` - Model training tests
- `tests/test_prediction_pipeline.py` - Prediction tests
- `tests/test_preprocessing_pipeline.py` - Preprocessing tests

## 📊 Monitoring and Logging

### Logging
- **Framework**: Loguru
- **Levels**: DEBUG, INFO, WARNING, ERROR, SUCCESS
- **Output**: Console and file logging
- **Format**: Structured JSON logging

### Metrics
- **Model Performance**: Accuracy, precision, recall, F1-score
- **System Metrics**: Response time, throughput, error rates
- **Business Metrics**: Prediction distribution, feature importance

### Health Checks
- **Model Health**: Model loading and prediction capability
- **Data Health**: Data quality and availability
- **System Health**: API responsiveness and resource usage

## 🔧 Configuration

### Environment Variables
```bash
# Data Configuration
DATA_SOURCE=s3  # or local
S3_BUCKET=nutrisage
S3_PREFIX=data/

# Model Configuration
MODEL_PATH=models/nutrition_grade_model.pkl
METADATA_PATH=models/model_metadata.pkl

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# AWS Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

### Configuration Files
- `config/preprocessing_config.json` - Preprocessing parameters
- `config/model_config.json` - Model hyperparameters
- `config/api_config.json` - API configuration

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Style
- **Python**: PEP 8 with Black formatting
- **Type Hints**: Use type hints for all functions
- **Documentation**: Docstrings for all classes and functions
- **Testing**: Minimum 80% code coverage

### Commit Convention
```
feat: add new feature
fix: bug fix
docs: documentation changes
test: add or update tests
refactor: code refactoring
style: formatting changes
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Open Food Facts**: For providing the nutritional data
- **XGBoost**: For the excellent gradient boosting implementation
- **FastAPI**: For the modern Python web framework
- **Docker**: For containerization technology

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@nutrisage.com

---

**Version**: 1.0.0  
**Last Updated**: [Current Date]  
**Maintainer**: NutriSage ML Team

