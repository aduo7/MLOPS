import requests

ride={"PULocationID":20,
      "DOLocationID":120,
      "trip_distance": 80}

url='http://127.0.0.1:9696/predict'

response=requests.post(url, json=ride)
print(response.json())