import pandas as pd

from django.http import JsonResponse
from rap import utils

def get_data(nama_coin):
    df = utils.get_coin(nama_coin)
    ts = pd.DataFrame(df,columns=['timestamp'])
    price = pd.DataFrame(df,columns=['last_price'])
    data = price.values
    data = data.astype('float32')
    timestamp = ts.values

    response = {}
    actual_price = []
    
    for x in range (len(data)-100, len(data)):
        temp_1 = {}
        temp_1["price"] = data[x][0].tolist()
        temp_1["timestamp"] = timestamp[x][0]
        actual_price.append(temp_1)

    response["actual_prices"] = actual_price
    
    return JsonResponse(response)