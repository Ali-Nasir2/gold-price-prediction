# 🏆 Gold Price Prediction using CNN-LSTM Hybrid Model

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated deep learning project that predicts gold prices using a hybrid CNN-LSTM neural network architecture. This model combines Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for time-series prediction.

## 📊 Project Overview

This project implements a state-of-the-art machine learning pipeline for gold price forecasting based on historical yearly data from 1978 to 2022. The model achieves impressive performance with a MAPE (Mean Absolute Percentage Error) of approximately 10.58% and a classification accuracy of 77.78% for predicting price direction.

### Key Features

- **Hybrid CNN-LSTM Architecture**: Combines the power of CNNs for spatial feature extraction with LSTMs for temporal pattern recognition
- **Comprehensive Data Preprocessing**: Log transformation and min-max normalization for improved model performance
- **Multiple Evaluation Metrics**: RMSE, MAE, MAPE for regression, and accuracy, sensitivity, specificity, AUC for classification
- **Visual Analytics**: Automated generation of insightful plots including price series, predictions, ROC curves, and training loss
- **Future Price Forecasting**: Predicts the next period's gold price based on historical patterns
- **Automated Reporting**: Generates comprehensive markdown reports with all model metrics and results

## 🎯 Model Performance

The model demonstrates robust performance across multiple metrics:

| Metric | Value |
|--------|-------|
| **RMSE** | 205.38 USD |
| **MAE** | 152.13 USD |
| **MAPE** | 10.58% |
| **Directional Accuracy** | 77.78% |
| **AUC Score** | 0.8571 |
| **Sensitivity** | 0.7143 |
| **Specificity** | 1.0000 |

## 🏗️ Model Architecture

```
Input Layer (Time Horizon: 4 steps)
    ↓
Conv1D (32 filters, kernel_size=2, activation='relu')
    ↓
Conv1D (64 filters, kernel_size=2, activation='relu')
    ↓
MaxPooling1D (pool_size=2)
    ↓
LSTM (100 units)
    ↓
Dense (1 output)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ali-Nasir2/gold-price-prediction.git
cd gold-price-prediction
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Or install all at once:
```bash
pip install numpy==1.21.0 pandas==1.3.0 matplotlib==3.4.2 scikit-learn==0.24.2 tensorflow==2.8.0
```

### Required Libraries

- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and CSV file handling
- **Matplotlib**: Data visualization and plotting
- **Scikit-learn**: Machine learning metrics and preprocessing
- **TensorFlow/Keras**: Deep learning framework for model building

## 💻 Usage

### Running the Model

Execute the main script to train the model and generate predictions:

```bash
python project1.py
```

### What Happens When You Run It?

1. **Data Loading**: Loads historical gold price data from `Yearly_Avg.csv`
2. **Preprocessing**: Applies log transformation and normalization
3. **Model Training**: Trains the CNN-LSTM model with early stopping
4. **Evaluation**: Calculates comprehensive performance metrics
5. **Visualization**: Generates multiple plots and charts
6. **Future Prediction**: Forecasts the next period's gold price
7. **Report Generation**: Creates a detailed markdown report

### Output Structure

After execution, a `gold_forecast_results/` directory is created with:

```
gold_forecast_results/
├── gold_price_cnn_lstm_model.h5      # Trained model file
├── gold_price_series.png              # Historical price visualization
├── training_loss.png                  # Training/validation loss curves
├── predictions.png                    # Actual vs predicted prices
├── direction_prediction.png           # Price direction accuracy visualization
├── roc_curve.png                      # ROC curve for classification
├── price_predictions.csv              # Detailed predictions data
├── classification_metrics.csv         # Classification metrics
├── future_prediction.txt              # Next period forecast
└── gold_price_prediction_report.md    # Comprehensive report
```

## 📁 Project Structure

```
gold-price-prediction/
│
├── project1.py                        # Main training and prediction script
├── Yearly_Avg.csv                     # Historical gold price dataset (1978-2022)
├── README.md                          # Project documentation
│
└── gold_forecast_results/             # Generated output directory
    ├── *.png                          # Visualization plots
    ├── *.csv                          # Prediction and metrics data
    ├── *.h5                           # Trained model
    ├── *.txt                          # Future predictions
    └── *.md                           # Generated reports
```

## 📈 Dataset Information

### Data Source
The dataset (`Yearly_Avg.csv`) contains yearly gold prices from 1978 to 2022 in multiple currencies.

### Key Characteristics:
- **Time Period**: 1978-2022 (45 years)
- **Frequency**: Yearly averages
- **Primary Currency**: USD
- **Data Points**: 45 observations
- **Format**: CSV with multiple currency columns

### Data Preprocessing Steps:
1. **Currency Parsing**: Removes commas from USD values and converts to numeric
2. **Log Transformation**: Applies natural logarithm to stabilize variance
3. **Normalization**: Min-max scaling to [0, 1] range
4. **Windowing**: Creates rolling windows of 4 time steps for sequence learning
5. **Train-Test Split**: 70% training, 30% testing

## 🔬 Methodology

### Time Series Approach
The model uses a supervised learning approach with rolling windows:
- **Window Size**: 4 previous time steps
- **Prediction**: Next time step
- **Training Strategy**: Sequential validation with early stopping

### Classification Task
In addition to price prediction, the model also predicts price movement direction (up/down):
- Binary classification (1 = price increase, 0 = price decrease)
- Useful for trading decisions and trend analysis

## 📊 Visualizations

The project generates several insightful visualizations:

1. **Gold Price Series**: Historical trend of gold prices over time
2. **Training Loss**: Model convergence during training
3. **Predictions Plot**: Comparison of actual vs predicted prices
4. **Direction Prediction**: Visual representation of correct/incorrect direction predictions
5. **ROC Curve**: Model performance for binary classification

## 🎓 Technical Details

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 16
- **Epochs**: 100 (with early stopping)
- **Early Stopping**: Patience of 20 epochs on validation loss
- **Validation**: Hold-out 30% test set

### Feature Engineering
- Rolling window approach for temporal dependencies
- Log transformation to handle exponential growth
- Min-max normalization for stable training

## 🔮 Future Predictions

The model can forecast future gold prices by:
1. Taking the last 4 time steps from the test set
2. Feeding them through the trained model
3. Inverse transforming to get the actual price prediction

Example output: `Next period gold price prediction: $1890.63`

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

- **Data Enhancement**: Add more frequent data (monthly/daily) for better predictions
- **Model Improvements**: Experiment with different architectures or hyperparameters
- **Feature Addition**: Incorporate additional economic indicators (inflation, interest rates, etc.)
- **Ensemble Methods**: Implement ensemble predictions for improved accuracy
- **UI Development**: Create a web interface for interactive predictions

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Implement multi-step ahead forecasting
- [ ] Add support for daily/monthly data
- [ ] Integrate external economic indicators
- [ ] Create web-based dashboard for real-time predictions
- [ ] Implement ensemble models (Random Forest, XGBoost, etc.)
- [ ] Add sentiment analysis from news/social media
- [ ] Develop API for prediction services
- [ ] Add backtesting framework for trading strategies

## 📚 References

This project implements concepts from:
- Time series forecasting using deep learning
- Hybrid CNN-LSTM architectures for sequential data
- Financial market prediction methodologies

## ⚠️ Disclaimer

**Important**: This model is for educational and research purposes only. Gold price predictions should not be used as the sole basis for investment decisions. Financial markets are influenced by numerous complex factors, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ali Nasir**

- GitHub: [@Ali-Nasir2](https://github.com/Ali-Nasir2)
- Repository: [gold-price-prediction](https://github.com/Ali-Nasir2/gold-price-prediction)

## 🌟 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Scikit-learn for machine learning utilities
- The open-source community for inspiration and support

## 📞 Contact & Support

If you have any questions, suggestions, or issues:
- Open an issue on GitHub
- Contact the author through GitHub profile
- Star ⭐ the repository if you find it useful!

---

<div align="center">
Made with ❤️ by Ali Nasir | © 2024
</div>
