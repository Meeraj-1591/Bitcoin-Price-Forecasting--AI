# Bitcoin Price Forecasting using Deep Neural Networks 🪙📈

**Author:** Budike Meeraj Kumar  


---

## Project Overview

This project predicts Bitcoin's next-day closing price using a **Feedforward Neural Network (FNN)**. By analyzing historical price data, creating lag and technical features, and leveraging nonlinear patterns, the model provides accurate forecasts to support investors, traders, and financial analysts in cryptocurrency markets.

---

## Business Problem

Bitcoin's highly volatile prices are influenced by market sentiment, trading range, and global economic events. Accurate forecasting is crucial for investors and institutions to optimize trading strategies, minimize losses, and make informed decisions in a fast-evolving digital market.

---

## ML Task Formulation

- **Problem Type:** Supervised Learning (Regression)  
- **Input:** Historical Bitcoin prices (past 60 days)  
- **Output:** Next-day closing price  
- **Model:** Feedforward Neural Network (FNN)  
- **Evaluation Metrics:** MAE, RMSE, R² score  
- **Dataset:** [Bitcoin Price Data](https://www.kaggle.com/code/abhishek14398/bitcoin-prediction-and-forecasting/input)  

---

## Key Steps

1. **Data Loading & EDA**  
   - Explored historical Bitcoin prices  
   - Visualized trends, volatility, and distribution  

2. **Data Preprocessing & Feature Engineering**  
   - Converted dates and sorted chronologically  
   - Created lag features, rolling averages, and technical indicators  
   - Defined target variable for next-day price  

3. **Train/Validation/Test Split & Scaling**  
   - 70% training, 15% validation, 15% testing  
   - Applied `MinMaxScaler` on features  

4. **Model Development**  
   - FNN with 3 hidden layers (128, 64, 32 neurons)  
   - ReLU activations, Dropout 0.2  
   - Output layer: Linear activation for regression  

5. **Training & Evaluation**  
   - Optimizer: Adam  
   - Epochs: 100, Batch size: 32  
   - Achieved Test MAE: 926.19, RMSE: 1520.01, R²: 0.9933  

---

## Results

- The model accurately predicts Bitcoin prices on the test set  
- Provides visualizations for **actual vs predicted prices**  
- Highlights the effectiveness of FNN for financial time-series forecasting  

---

## Usage

```python
# Clone the repository
git clone https://github.com/Meeraj-1591/YouTube-Comments-Sentiment-Analysis.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Bitcoin_FNN_Prediction.ipynb
