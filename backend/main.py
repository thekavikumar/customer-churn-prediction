# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
from customer import Customer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

@app.post('/predict')
def predict(data: Customer):
    # input_data = pd.DataFrame([data], columns=[
    #     'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    #     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    #     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 
    # ])
    # num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    # scaler= StandardScaler()
    # if 'customerID' in input_data.columns:
    #     input_data.drop('customerID', axis=1, inplace=True)
    # input_data = input_data.apply(lambda x: object_to_int(x))
    # input_data[num_cols] = scaler.transform(input_data[num_cols])
    # prediction = model.predict(input_data)
    # if prediction[0] == 1:
    #     result = "Churn"
    # else:
    #     result = "No Churn"

    return {
        'prediction': data
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload