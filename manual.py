from time import sleep
import requests

#ENDPOINT = "http://localhost:7071/api/CorineHttpTrigger"
ENDPOINT = "https://landcover-demo-2.azurewebsites.net/api/CorineHttpTrigger"

payloads = [
    {
        "lat":52.688803,
        "lon": -7.827722,
        "id": "francesc@odostech.com",
        "firstname": "Kevin"
    },
]

for payload in payloads:
    req = requests.get(
        ENDPOINT,
        params=payload
    )

    print(req.text)
    sleep(10)