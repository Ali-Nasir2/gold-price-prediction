# Gold Price Prediction Project Report
Generated on 2025-05-12 15:50

## Model Implementation
This project implements a CNN-LSTM hybrid model for gold price prediction based on yearly data.
The model architecture follows the CNN-LSTM design from the paper, with:
- Two convolutional layers (32 and 64 filters)
- Max pooling layer
- LSTM layer with 100 units
- Dense output layer

## Data Description
- Dataset: Yearly gold prices
- Time horizon: 4 time steps
- Train/Test split: 70%/30%
- Data preprocessing: Log transformation and normalization to (0,1) range

## Model Performance
- RMSE: 205.38
- MAE: 152.13
- MAPE: 10.58%

## Classification Performance
- Accuracy: 77.78%
- Sensitivity: 0.7143
- Specificity: 1.0000
- AUC: 0.8571

## Future Prediction
- Next period gold price: $1890.63

## Files Generated
- Model file: gold_price_cnn_lstm_model.h5
- Gold price series visualization: gold_price_series.png
- Training loss plot: training_loss.png
- Predictions plot: predictions.png
- Classification visualization: direction_prediction.png (if available)
- ROC curve: roc_curve.png (if available)

## Model File Location
Saved model file: C:\Users\alina\OneDrive\Desktop\ai as\gold_forecast_results\gold_price_cnn_lstm_model.h5

## Next Steps
- Try using monthly or daily data to improve classification metrics
- Test different time horizons or model architectures
- Implement ensemble predictions for higher accuracy
