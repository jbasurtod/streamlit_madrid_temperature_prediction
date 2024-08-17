# Predicting Temperatures in Barrio San Isidro, Madrid



## About

This project showcases a productized XGBoost time series model, trained with historical temperature data from Madrid's open data portal. The model is used to predict hourly temperatures in Barrio San Isidro, Madrid, and is deployed in a Streamlit app. You can explore the app [here](https://predicting-madrid-temp.streamlit.app/).

This repository contains the code for the Streamlit app that presents the model’s predictions and performance.

## Project Structure

The project involved the following key components:

### 1. Model Training

The model was trained using historical temperature data obtained from [Madrid's open data portal](https://datos.madrid.es/sites/v/index.jsp?vgnextoid=fa8357cec5efa610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD). A detailed Kaggle Notebook explaining the model training process will be uploaded soon.

### 2. ETL Pipeline

The data pipeline was implemented using Apache Airflow and involved the following steps:
- **Extract**: Hourly temperature data is fetched from Madrid's open data portal. The data is updated around the 30th minute of every hour.
- **Transform**: The latest temperature data is integrated with historical data to create a comprehensive dataset. This dataset is then used to make predictions for the next hour and the next 24 hours.
- **Load**: The predictions and the actual temperatures are stored in Google Drive. The predictions are evaluated using the Mean Absolute Percentage Error (MAPE) to measure model performance.

### 3. Streamlit App

The Streamlit app displays:
- The last 20 hours of actual vs. predicted temperature data to assess the model’s recent performance.
- Temperature predictions for the next 24 hours.

## Getting Started

Follow the steps below to clone the repository and run the Streamlit app locally.

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.7 or later
- Git

### Installation

1. **Clone the repository**:
   ```bash
    git clone https://github.com/jbasurtod/streamlit_madrid_temperature_prediction.git
    cd streamlit_madrid_temperature_prediction
    ```
2. **Create and activate a virtual environment:**
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. **Install the required dependencies:**
   ```bash
    pip install -r requirements.txt
    ```
4. **Running the App:**
   ```bash
    streamlit run app.py
    ```
This will start a local server, and you can view the app in your browser at http://localhost:8501.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## Acknowledgements

- Thanks to [Madrid's open data portal](https://datos.madrid.es) for providing the data.
- The model was built using [XGBoost](https://xgboost.readthedocs.io/), a powerful gradient boosting library.
- The app was developed using [Streamlit](https://streamlit.io/), a framework for building data apps.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
