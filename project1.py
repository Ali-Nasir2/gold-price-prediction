import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, InputLayer

results_dir = "gold_forecast_results"
os.makedirs(results_dir, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(results_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved {name}.png to {results_dir}")

print("\nLoading and Processing Data ")
df = pd.read_csv('Yearly_Avg.csv')
print('Dataset shape:', df.shape)
print(df.dtypes)
print(df.head())

df['USD'] = pd.to_numeric(df['USD'].str.replace(',', ''), errors='coerce')
gold_prices = df['USD'].values
valid_prices = gold_prices[~np.isnan(gold_prices)]
print(f"Min: {valid_prices.min()}, Max: {valid_prices.max()}, Mean: {valid_prices.mean()}")

plt.figure(figsize=(12, 6))
plt.plot(valid_prices)
plt.xlabel("Years")
plt.ylabel("Price (USD)")
plt.title("Gold Prices Over Time")
plt.grid()
save_fig("gold_price_series")
plt.show()

log_prices = np.log(valid_prices)

train_size = int(len(log_prices) * 0.7) 
train_set = log_prices[:train_size]
test_set = log_prices[train_size:]
print(f"Train: {train_set.shape}, Test: {test_set.shape}")

def rolling_window_mtx(x, window_size):
    """Compute overlapping observation windows over a vector"""
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > x.shape[0]:
        raise ValueError("`window_size` is too long.")

    shape = (x.shape[0] - window_size + 1, window_size)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


time_horizon = 4
x_train = rolling_window_mtx(train_set, time_horizon)[:-1]
y_train = rolling_window_mtx(train_set[time_horizon:], 1)

x_test = rolling_window_mtx(test_set, time_horizon)[:-1]
y_test = rolling_window_mtx(test_set[time_horizon:], 1)

print(f"Training data shape: X={x_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={x_test.shape}, y={y_test.shape}")

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train = scaler_x.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train)
x_test = scaler_x.transform(x_test)
y_test = scaler_y.transform(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print("\nBuilding and Training Model")
model = Sequential()
model.add(InputLayer(input_shape=(time_horizon, 1)))
model.add(Conv1D(32, 2, activation='relu', padding='same'))
model.add(Conv1D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='valid'))
model.add(LSTM(100))
model.add(Dense(1))

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
save_fig("training_loss")
plt.show()

print("\n=== Model Evaluation ===")
y_pred = model.predict(x_test)

y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

y_test_real = np.exp(y_test_inv)
y_pred_real = np.exp(y_pred_inv)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f'Regression Metrics:')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label='Actual Price')
plt.plot(y_pred_real, label='Predicted Price')
plt.title('Gold Price Prediction (Test Set)')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
save_fig("predictions")
plt.show()

prices_df = pd.DataFrame({
    'Actual': y_test_real.flatten(),
    'Predicted': y_pred_real.flatten()
})
prices_df.to_csv(os.path.join(results_dir, "price_predictions.csv"), index=False)

print("\nClassification Analysis")

def direction_classify(data):
    """Convert price data to binary classification (up=1, down=0)"""
    directions = []
    for i in range(1, len(data)):
        directions.append(1 if data[i] > data[i-1] else 0)
    return np.array(directions)

if len(y_test_real) > 1:
    actual_directions = direction_classify(y_test_real.flatten())
    pred_directions = direction_classify(y_pred_real.flatten())
    
    if len(actual_directions) > 0 and len(pred_directions) > 0:
        min_len = min(len(actual_directions), len(pred_directions))
        actual_directions = actual_directions[:min_len]
        pred_directions = pred_directions[:min_len]
        
        accuracy = accuracy_score(actual_directions, pred_directions) * 100
        conf_matrix = confusion_matrix(actual_directions, pred_directions)
        
        if conf_matrix.size == 4:  
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            try:
                fpr, tpr, _ = roc_curve(actual_directions, pred_directions)
                auc_value = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve for Price Direction Prediction')
                plt.legend(loc='lower right')
                plt.grid(True)
                save_fig("roc_curve")
                plt.show()
            except Exception as e:
                print(f"Could not calculate ROC curve: {e}")
                auc_value = None
                
            print("\nClassification Metrics (Price Direction):")
            print(f"Accuracy: {accuracy:.2f}%")
            if auc_value:
                print(f"AUC: {auc_value:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_real, label='Actual Prices')
            plt.plot(y_pred_real, label='Predicted Prices', linestyle='--')
            
            for i in range(min_len):
                marker_position = i+1  
                if marker_position < len(y_test_real):
                    if actual_directions[i] == pred_directions[i]:
                        plt.scatter(marker_position, y_test_real[marker_position], color='green', s=100, zorder=3)  # Correct
                    else:
                        plt.scatter(marker_position, y_test_real[marker_position], color='red', s=100, zorder=3)  # Wrong
            
            plt.title('Gold Price Prediction with Direction Accuracy')
            plt.xlabel('Time Steps')
            plt.ylabel('Price (USD)')
            plt.legend(['Actual Prices', 'Predicted Prices', 'Correct Direction', 'Wrong Direction'])
            plt.grid(True)
            save_fig("direction_prediction")
            plt.show()
            
            class_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'AUC'],
                'Value': [f"{accuracy:.2f}%", f"{sensitivity:.4f}", f"{specificity:.4f}", 
                          f"{auc_value:.4f}" if auc_value else "N/A"]
            })
            class_df.to_csv(os.path.join(results_dir, "classification_metrics.csv"), index=False)
        else:
            print("Classification metrics could not be calculated ")
    else:
        print("Not enough points to calculate arrays")
else:
    print("Not enough test points for classification")

model_path = os.path.join(results_dir, "gold_price_cnn_lstm_model.h5")
model.save(model_path)
print(f"\nModel saved to: {os.path.abspath(model_path)}")

print("\nFuture Price Prediction")
try:
    if len(test_set) >= time_horizon:
        future_input = test_set[-time_horizon:].reshape(1, -1)
        future_input_scaled = scaler_x.transform(future_input)
        future_input_reshaped = future_input_scaled.reshape(1, time_horizon, 1)
        
        future_pred_scaled = model.predict(future_input_reshaped)
        future_pred_inv = scaler_y.inverse_transform(future_pred_scaled)
        future_pred = np.exp(future_pred_inv)[0][0]
        
        print(f"Next period prediction: ${future_pred:.2f}")
        
        with open(os.path.join(results_dir, "future_prediction.txt"), "w") as f:
            f.write(f"Next period gold price prediction: ${future_pred:.2f}")
            f.write(f"\nPrediction date: {datetime.now().strftime('%Y-%m-%d')}")
    else:
        print("Not enough data for future prediction")
except Exception as e:
    print(f"Error in future prediction: {e}")

print("\nGenerating Final Report")

report_content = f"""# Gold Price Prediction Project Report
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Model Implementation
This project implements a CNN-LSTM hybrid model for gold price prediction based on yearly data.
The model architecture follows the CNN-LSTM design from the paper, with:
- Two convolutional layers (32 and 64 filters)
- Max pooling layer
- LSTM layer with 100 units
- Dense output layer

## Data Description
- Dataset: Yearly gold prices
- Time horizon: {time_horizon} time steps
- Train/Test split: 70%/30%
- Data preprocessing: Log transformation and normalization to (0,1) range

## Model Performance
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MAPE: {mape:.2f}%

## Classification Performance
"""

if 'accuracy' in locals():
    auc_text = f"{auc_value:.4f}" if 'auc_value' in locals() and auc_value is not None else "N/A"
    report_content += f"""- Accuracy: {accuracy:.2f}%
- Sensitivity: {sensitivity:.4f}
- Specificity: {specificity:.4f}
- AUC: {auc_text}
"""
else:
    report_content += "Classification metrics could not be calculated due to insufficient data.\n"

future_pred_text = "N/A"
if 'future_pred' in locals():
    future_pred_text = f"${future_pred:.2f}"

report_content += f"""
## Future Prediction
- Next period gold price: {future_pred_text}

## Files Generated
- Model file: {os.path.basename(model_path)}
- Gold price series visualization: gold_price_series.png
- Training loss plot: training_loss.png
- Predictions plot: predictions.png
- Classification visualization: direction_prediction.png (if available)
- ROC curve: roc_curve.png (if available)

## Model File Location
Saved model file: {os.path.abspath(model_path)}

## Next Steps
- Try using monthly or daily data to improve classification metrics
- Test different time horizons or model architectures
- Implement ensemble predictions for higher accuracy
"""

report_path = os.path.join(results_dir, "gold_price_prediction_report.md")
with open(report_path, "w") as f:
    f.write(report_content)

print(f"Comprehensive report saved to: {os.path.abspath(report_path)}")
print(f"\nAll results have been saved to: {os.path.abspath(results_dir)}")
print("=== Gold Price Forecasting Project Completed ===")
