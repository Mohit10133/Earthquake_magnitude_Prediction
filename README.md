# 🌍 Earthquake Magnitude Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

> 🎯 An advanced machine learning project that predicts earthquake magnitudes using seismic and geographical features, powered by XGBoost and presented through an interactive Streamlit web application.

If you find this project useful, please consider giving it a ⭐ star. It helps make the project more visible and encourages continued development!

## 📁 Project Structure

📊 `ML_PROJECT_FINAL.ipynb`
- Complete data analysis pipeline
- Model development and training
- Performance evaluation and visualization
- Detailed documentation and insights

🚀 `app.py`
- Interactive Streamlit web application
- Real-time earthquake magnitude predictions
- User-friendly interface

🤖 `earthquake_xgb_mag.pkl`
- Trained XGBoost model
- Optimized for accurate predictions
- Ready for deployment

📈 `Earthquake.csv`
- Rich dataset with seismic information
- Historical earthquake records
- Multiple geographical and technical features

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

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| 🐍 Core | Python 3.x |
| 📊 Data Processing | Pandas, NumPy |
| 🤖 Machine Learning | Scikit-learn, XGBoost |
| 🌐 Web App | Streamlit |
| 📈 Visualization | Matplotlib, Seaborn |

## 👨‍💻 Author

Created with 💖 by [Mohit10133](https://github.com/Mohit10133)

---

### 🌟 Support This Project

If you found this project helpful or interesting, please consider:
- Giving it a star ⭐
- Sharing it with others who might find it useful
- Contributing to its improvement

Your support helps maintain and improve this project! Thank you! 🙏
