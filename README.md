# Earthquake Magnitude Prediction

This repository contains a machine learning project that predicts earthquake magnitudes using various seismic and geographical features. The project includes both a machine learning model and a web application for making predictions.

## Project Structure

- `ML_PROJECT_FINAL.ipynb`: Jupyter notebook containing the complete data analysis, model development, and evaluation pipeline
- `app.py`: Streamlit web application for making earthquake magnitude predictions
- `earthquake_xgb_mag.pkl`: Trained XGBoost model saved as a pickle file
- `Earthquake.csv`: Dataset containing earthquake information including features like location, magnitude, alerts, and various seismic measurements

## Dataset Description

The dataset (`Earthquake.csv`) contains comprehensive earthquake data with the following key features:

- `magnitude`: The magnitude of the earthquake (target variable)
- `cdi`: Community Decimal Intensity
- `mmi`: Modified Mercalli Intensity
- `alert`: Alert level (encoded: green, yellow, etc.)
- `sig`: Significance of the earthquake
- `nst`: Number of stations that reported the event
- `dmin`: Minimum distance to stations
- `gap`: Largest azimuthal gap between azimuthally adjacent stations
- `magType`: Type of magnitude measurement
- `depth`: Depth of the earthquake
- `latitude`: Latitude of the epicenter
- `longitude`: Longitude of the epicenter

## Machine Learning Model

The project uses an XGBoost regression model to predict earthquake magnitudes. The model development process includes:

1. Data preprocessing and cleaning
2. Feature engineering
3. Model training and hyperparameter tuning
4. Model evaluation using metrics like RMSE and R² score
5. Model serialization for deployment

## Web Application

The project includes a Streamlit web application (`app.py`) that provides a user-friendly interface for making earthquake magnitude predictions. The application:

- Loads the trained XGBoost model
- Processes user input using the same preprocessing steps as during training
- Provides real-time predictions of earthquake magnitude

### Running the Application

1. Install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn xgboost
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Files

- `earthquake_xgb_mag.pkl`: Serialized XGBoost model file that contains the trained model for making predictions

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Run the Jupyter notebook to see the complete analysis and model development process
4. Launch the web application to make predictions

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Author

Mohit10133

## License

This project is open source and available under the MIT License.
