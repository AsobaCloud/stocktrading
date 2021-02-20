from Stock_List import Access_Tickers
from Get_Data import GetData
import pandas as pd
from google.cloud import aiplatform
from google.cloud import bigquery
import os 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './asoba-stocktrading-ed899163cfdc.json'

def predict_custom_model_sample(endpoint: str, instances_list: list, parameters_dict: dict):
    client_options = dict(api_endpoint="us-central1-prediction-aiplatform.googleapis.com")
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    # The format of the parameters must be consistent with what the model expects.
    parameters = json_format.ParseDict(parameters_dict, Value())

    # The format of the instances must be consistent with what the model expects.
    # instances_list = [instance]
    instances = [json_format.ParseDict(s, Value()) for s in instances_list]
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    predictions = response.predictions
    print("predictions")
    # for prediction in predictions:
    #     print(" prediction:", dict(prediction))
        
    return predictions


def todays_stock_data():
    tickers = Access_Tickers().get_stocks()
    data_frames = []
    for ticker in tickers[:15]:
        t = GetData(ticker, True).prepare_data_for_training()
        t['ticker'] = ticker
        # Date and short_result are not needed
        # t = t[t.columns.drop('date')]
        t = t[t.columns.drop('short_result')]
        data_frames.append(t)
    return pd.concat(data_frames),tickers


def preprocess_df(df):
    df['KAMA_20_period_KAMA_'] = df['KAMA_20_period_KAMA.']
    df['WMA_9_period_WMA_'] = df['WMA_9_period_WMA.']
    df['HMA_16_period_HMA_'] = df['HMA_16_period_HMA.']
    df['EVWMA_20_period_EVWMA_'] = df['EVWMA_20_period_EVWMA.']
    df['VWAP_VWAP_'] = df['VWAP_VWAP.']
    df.drop('KAMA_20_period_KAMA.', axis=1 , inplace=True)
    df.drop('WMA_9_period_WMA.', axis=1 , inplace=True)
    df.drop('HMA_16_period_HMA.', axis=1 , inplace=True)
    df.drop('EVWMA_20_period_EVWMA.', axis=1 , inplace=True)
    df.drop('VWAP_VWAP.', axis=1 , inplace=True)
    return


if __name__ == "__main__": 
    
    model_uri= "projects/287706207605/locations/us-central1/endpoints/2633937278942052352"
    df,tickers = todays_stock_data()
    preprocess_df(df)
    df = df.applymap(str)
    d = df.to_dict('records')
    # params = d[1]
    # ticker = params['ticker']
    # del params['ticker']
    predictions = predict_custom_model_sample(model_uri ,d, {})
    value = predictions[0]['value']
 
    for params,value in zip(d,predictions):
        # print(params, value['value'])
        params['value'] = value['value']
    
    data_ = d
    client = bigquery.Client()
    table_id = "model_2"
    dataset_id = "stocks_ml"
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)   
    client.insert_rows( table, data_, skip_invalid_rows= False, ignore_unknown_values=False)


