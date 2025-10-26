import logging
logging.Logger.root.level = 10

import os
import tempfile
import uuid
import threading
import requests
import json
import datetime


import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

#from azure.identity import DefaultAzureCredential
#from azure.keyvault.secrets import SecretClient

import wine_demo_service
from config import Config



app = func.FunctionApp()


#config = Config()
#AZURE_STORAGE_CONNECTION_STRING = config.get_string('AZURE_STORAGE_CONNECTION_STRING')
#AZURE_STORAGE_CONTAINER_NAME = config.get_string('AZURE_STORAGE_CORINE_CONTAINER_NAME')
#CALLBACK_URL = config.get_string("CALLBACK_URL")

# Initialize the client with DefaultAzureCredential
"""
client = SecretClient(
    vault_url="https://kviadevnortheurope-001.vault.azure.net",
    credential=DefaultAzureCredential()
)

AZURE_STORAGE_CONNECTION_STRING = client.get_secret('AZURE-STORAGE-CONNECTION-STRING').value
"""

AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nonameforyou;AccountKey=nokeyforyou;EndpointSuffix=core.windows.net"
AZURE_STORAGE_CONTAINER_NAME = 'fzc-corine-demo'
CALLBACK_URL = 'webhookhere.com'


#print(AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, CALLBACK_URL)
# { "lat": 43.37501473686352, "lon": -5.294759607267329 }



@app.route(route="CallbackDemo", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def callback_demo(req: func.HttpRequest) -> func.HttpResponse:
    """Dummy callback endpoint for testing the webhook POSTs.

    It logs the JSON body if present, otherwise logs the raw body bytes.
    """
    try:
        body = req.get_json()
        #print(body)
        logging.info(f"CallbackDemo received JSON: {json.dumps(body)}")
    except ValueError:
        raw = req.get_body()
        try:
            text = raw.decode('utf-8')
            logging.info(f"CallbackDemo received body: {text}")
        except Exception:
            logging.info(f"CallbackDemo received raw bytes: {raw}")

    return func.HttpResponse(f'Callback received {datetime.datetime.now()}', status_code=200)





def _upload_blob(src_path, blob_name):
    # Create client and ensure container exists (Azurite/local dev storage doesn't auto-create)
    client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

    blob = container_client.get_blob_client(
        #blob=str(uuid.uuid4()) + ".png"
        blob=blob_name
    )

    with open(src_path, 'rb') as data:
        blob.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="png")
        )

    return blob.url

def _process_and_callback(lat, lon, name, callback_url, task_id, demo_ID):
    # Background worker: generate climate analysis, upload results, and POST to callback_url.

    logging.info(f"[{task_id}] Background task started for: {demo_ID} | {name} -> {lat},{lon}")

    try:
        # Use the new climate analysis service
        results = wine_demo_service.generate_landcover_report(lat=lat, lon=lon, id=demo_ID, firstname=name)

        if results.get('success'):
            # Upload the generated visualization to blob storage
            visualization_path = results.get('visualization_path')
            blob_name = f"{demo_ID}-climate_analysis.png"
            
            if visualization_path and os.path.exists(visualization_path):
                visualization_url = _upload_blob(visualization_path, blob_name)
            else:
                raise Exception("Visualization file not found")

            payload = {
                'firstname': name,
                'demoID': demo_ID,
                'status': 'success',
                'coordinates': results.get('coordinates'),
                'scenarios_processed': results.get('scenarios_processed'),
                'climate_visualization': visualization_url
            }
        else:
            # Handle error case
            raise Exception(results.get('error', 'Unknown error in climate analysis'))


        logging.info(f"[{task_id}] Sending payload: {payload}")
        

        # POST results to callback
        try:
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(callback_url, json=payload, headers=headers, timeout=10)
            logging.info(f"[{task_id}] Callback POST status {resp.status_code}")
        except Exception as e:
            logging.exception(f"[{task_id}] Error: failed to POST to callback {callback_url}: {e}")

    except Exception as e:
        logging.exception(f"[{task_id}] Error: processing failed: {e}")

        # notify callback about failure
        try:
            payload = {
                'demoID': demo_ID,
                'status': 'failed',
                'error': str(e),
                'firstname': name
            }
            headers = {'Content-Type': 'application/json'}
            requests.post(callback_url, json=payload, headers=headers, timeout=10)
        except Exception:
            logging.exception(f"[{task_id}] Error: failed to POST failure to callback")
    finally:
        logging.info(f"[{task_id}] Background task finished")



@app.route(route="CorineHttpTrigger", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def corine(req: func.HttpRequest, context) -> func.HttpResponse:
    logging.info('CorineHttpTrigger: HTTP trigger received a request.')

    try: body = req.get_json()
    except ValueError: body = None

    lat = req.params.get('lat') or (body and body.get('lat'))
    lon = req.params.get('lon') or (body and body.get('lon'))
    name = req.params.get('firstname') or (body and body.get('firstname'))

    if lat is None or lon is None:
        logging.error(f'lat or lon is missing: {lat}, {lon}')
        return func.HttpResponse(json.dumps({'error': 'Please pass lat and lon in query string or JSON body'}), status_code=400, mimetype='application/json')

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception as e:
        logging.error(f'invalid lat or lon: {lat}, {lon}')
        return func.HttpResponse(json.dumps({'error': 'lat and lon must be numbers', 'details': str(e)}), status_code=400, mimetype='application/json')
    
    demo_ID = req.params.get('id') or (body and body.get('id'))
    if demo_ID is None:
        logging.error(f'demoid is missing: {demo_ID}')
        return func.HttpResponse(json.dumps({'error': 'the demo id is missing'}), status_code=400, mimetype='application/json')

    #callback_url = "https://hooks.zapier.com/hooks/catch/21440442/u9vi985/"
    #callback_url = "http://localhost:7071/api/CallbackDemo"

    # If a callback URL is provided, run processing in background and return 202 Accepted
    task_id = str(uuid.uuid4())
    
    """thread = threading.Thread(
        target=_process_and_callback,
        args=(lat, lon, name, CALLBACK_URL, task_id, demo_ID, context),
        daemon=True
    )
    
    thread.start()"""

    _process_and_callback(lat, lon, name, CALLBACK_URL, task_id, demo_ID)


    resp = {
        'task_id': task_id,
        'status': 'accepted',
        'message': 'Processing started; results will be POSTed to callback_url'
    }
    
    return func.HttpResponse(json.dumps(resp), status_code=202, mimetype='application/json')
