#  California Real Estate Price Predictor: End-to-End ML Solution

##  Project Overview
Accurate real estate valuation is a critical challenge for investors, brokers, and homebuyers. This project is an **End-to-End Machine Learning Solution** designed to predict housing prices in California based on various demographic and property-specific features. 

Rather than just building a predictive model in a notebook, this project culminates in a **fully interactive web application** that allows end-users to input property specifications and instantly receive a fair market value estimate.

##  Key Features & Business Value
* **Exploratory Data Analysis (EDA):** Deep dive into spatial data, identifying hidden correlations between geographical location, median income, and housing prices.
* **Model Explainability:** Uncovering the "Why" behind the price. The model provides feature importance analysis, revealing that `ocean_proximity` and `median_income` are the strongest drivers of property value.
* **Robust ML Pipeline:** Evaluated and tuned multiple algorithms (Random Forest, SVM, Gradient Boosting, XGBoost) to ensure high prediction reliability without overfitting.
* **Interactive Web Dashboard:** Deployed a user-friendly, responsive Streamlit application equipped with interactive maps and dynamic parameter adjustments.

##  Model Performance & Evaluation
After testing various algorithms, **XGBoost** outperformed the others, providing a highly stable and accurate prediction curve.
* **R² Score (Test):** `0.8404` (Explaining 84% of the variance in housing prices).
* **Mean Absolute Error (MAE):** `~$28,901`
* **Root Mean Squared Error (RMSE):** `~$45,734`

##  Technical Stack
* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment & UI:** Streamlit

##  Project Structure (Workflow)
1. **Data Preprocessing:** Handling missing values, log-transforming skewed features (like house values), and encoding categorical variables.
2. **Correlation Analysis:** Utilizing heatmaps to identify multicollinearity and select the most impactful features.
3. **Model Training & Comparison:** Benchmarking models against each other to find the optimal balance between accuracy and computational efficiency.
4. **App Deployment:** Wrapping the best-performing XGBoost model into a Streamlit script for real-time inference.on this link
   ```bash
   https://california-house-price-predictor-qgh9aonqbtukt6d6ytnwgd.streamlit.app/

## How to Run the App Locally
To run the interactive web application on your local machine, follow these steps:

1. Clone this repository:
   ```bash
   git clone [https://github.com/AdhamKhider/california-housing-predictor.git](https://github.com/AdhamKhider/california-housing-predictor.git)


   
