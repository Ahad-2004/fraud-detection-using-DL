# Professional Insurance Fraud Detection System

![Fraud Detection](https://img.freepik.com/free-vector/cyber-security-concept_23-2148543473.jpg?w=900&t=st=1700000000~exp=1700000600~hmac=abcdef123456)

## Overview
This project is a sophisticated insurance fraud detection system that uses deep learning to identify potentially fraudulent claims. It features:

- 🧠 **Deep Learning Model**: TensorFlow/Keras neural network with 97% validation accuracy
- 💻 **Modern Web Interface**: Beautiful HTML dashboard with real-time risk assessment
- ⚙️ **Robust Backend**: FastAPI server with comprehensive error handling
- 📊 **Data Preprocessing**: Advanced sklearn pipeline with feature engineering

## Key Features
- **Real-time Fraud Prediction**: Instant analysis of insurance claims
- **Risk Visualization**: Animated risk meter with color-coded threat levels
- **Professional UI**: Modern dashboard with intuitive form sections
- **API Endpoints**: Well-documented REST API for integration
- **Error Resilience**: Graceful handling of model loading failures

## Technology Stack
| Component | Technology |
|-----------|------------|
| **Frontend** | HTML5, Bootstrap 5, JavaScript, CSS3 |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI/ML** | TensorFlow/Keras, Scikit-Learn, Pandas |
| **Deployment** | GitHub, Local Server |

## Installation
```bash
# Clone repository
git clone https://github.com/Ahad-2004/fraud-detection-using-DL.git
cd fraud-detection-using-DL

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

## Usage
1. Start the server: `python main.py`
2. Open `http://localhost:8000` in your browser
3. Fill in the insurance claim details
4. Click "Analyze Fraud Risk"
5. View the risk assessment with confidence level

## Project Structure
```
fraud-detection-using-DL/
├── main.py               # FastAPI backend server
├── index.html            # Professional frontend UI
├── professional_fraud_model.h5    # Trained Keras model
├── professional_preprocessor.pkl  # Feature preprocessing pipeline
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (this file)
├── start_server.bat      # Windows server startup script
└── test_preprocessor.py  # Preprocessor validation script
```

## Results Interpretation
| Risk Level | Confidence | Action |
|------------|------------|--------|
| 🟢 **Very Low** | < 15% | Fast-track processing |
| 🟢 **Low** | 15-30% | Routine processing |
| 🟠 **Moderate** | 30-50% | Standard review |
| 🟡 **Elevated** | 50-75% | Detailed review |
| 🔴 **High** | 75-90% | Priority investigation |
| 🚨 **Extreme** | > 90% | Immediate action |

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- TensorFlow team for the deep learning framework
- Scikit-Learn for preprocessing tools
- FastAPI for the efficient web server
- Bootstrap for the responsive UI components
