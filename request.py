import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={ 'CYLINDERS':4, 'ENGINESIZE':2.4, 'FUELCONSUMPTION_CITY':11.2, 'FUELCONSUMPTION_HWY':7.7, 'FUELCONSUMPTION_COMB':9.6})

print(r.json())
