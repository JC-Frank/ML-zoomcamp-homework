import requests

url = "http://localhost:9696/predict"

patient = {
    "gender": "F",
    "sms_received": 1,
    "hipertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handcap": 0,
    "scholarship": 1,
    "neighbourhood": "JARDIM DA PENHA",
    "age": 37,
    "AppointmentDay": "2027-02-05T00:00:00Z"
    "ScheduledDay": "2026-11-14T18:38:08Z"
}

response = requests.post(url, json=patient).json()
print(response)

