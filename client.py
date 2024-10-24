import requests

body = {
    "Age": 30,
    "Gender": "Female",
    "Education": 1,  
    "IntroversionScore": 10.1647,
    "SensingScore": 8.1414337690409,
    "ThinkingScore": 4.03696,
    "JudgingScore": 5.36027834724605,
    "Interest": "Sports"
}

response = requests.post(url='http://127.0.0.1:8000/score', json=body)
print(response.json())
