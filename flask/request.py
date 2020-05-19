import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'preg':6, 'mass':33.6, 'age':50,'plas':148,'pres':72,'pedi':0.627})

print(r.json())
