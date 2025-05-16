# Customer Churn Prediction App

This Streamlit application predicts customer churn for a telecommunications company based on various customer attributes. The model helps identify customers who are at risk of canceling their service, allowing companies to take proactive retention measures.

## Features

- **Single Prediction**: Input details for an individual customer to predict their churn risk
- **Batch Prediction**: Upload a CSV file of multiple customers for bulk analysis
- **Model Information**: Learn about the prediction model and important factors affecting churn

## Demo

Visit the live app: [Customer Churn Prediction](https://your-streamlit-cloud-url-here)

![App Screenshot](https://example.com/screenshot.png)

## How It Works

The application uses a machine learning model (RandomForest classifier) to predict customer churn based on:

1. Customer demographics (gender, age, partner status)
2. Account information (tenure, contract type, billing method)
3. Services subscribed (phone, internet, add-ons)

## Local Development

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run churn_prediction_app.py
```

## Deployment

See the [deployment guide](deployment_guide.md) for instructions on deploying this app to Streamlit Cloud with Dropbox model integration.

## Data Source

The model was trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

## License

[MIT](LICENSE)
