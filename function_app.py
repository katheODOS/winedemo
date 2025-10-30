import logging
import sys

logging.getLogger().setLevel(logging.INFO)

import os
import tempfile
import uuid
import threading
import requests
import json
import datetime

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError, ServiceRequestError
from azure.storage.blob._models import BlobProperties

from wine_demo_service import ClimateAnalysisService

app = func.FunctionApp()

# Load environment variables
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", 
                                                  "DefaultEndpointsProtocol=https;AccountName=nonameforyou;AccountKey=nokeyforyou;EndpointSuffix=core.windows.net")
AZURE_STORAGE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_CONTAINER_NAME", 'winedemo2')
CALLBACK_URL = os.environ.get("CALLBACK_URL", 'webhookhere.com')

# CDS API credentials
CDS_API_URL = os.environ.get("CDS_API_URL")
CDS_API_KEY = os.environ.get("CDS_API_KEY")

# Validate CDS credentials
if not CDS_API_URL or not CDS_API_KEY:
    logging.warning("CDS API credentials not found in environment variables!")

@app.route(route="CallbackDemo", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def callback_demo(req: func.HttpRequest) -> func.HttpResponse:
    """Dummy callback endpoint for testing the webhook POSTs."""
    try:
        body = req.get_json()
        logging.info(f"CallbackDemo received JSON: {json.dumps(body)}")
    except ValueError:
        raw = req.get_body()
        try:
            text = raw.decode('utf-8')
            logging.info(f"CallbackDemo received body: {text}")
        except Exception:
            logging.info(f"CallbackDemo received raw bytes: {raw}")

    return func.HttpResponse(f'Callback received {datetime.datetime.now()}', status_code=200)


@app.route(route="StorageTest", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def storage_test(req: func.HttpRequest) -> func.HttpResponse:
    """Test endpoint to verify Azure Storage connectivity."""
    logging.info("StorageTest: Testing Azure Storage connectivity")
    
    # Test creating a small test blob
    try:
        test_content = b"Storage connectivity test"
        test_blob_name = f"test_connectivity_{datetime.datetime.now().isoformat()}.txt"
        
        client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        
        # Ensure container exists
        try:
            container_client.create_container()
        except ResourceExistsError:
            pass
        
        # Upload test blob
        blob_client = container_client.get_blob_client(blob=test_blob_name)
        blob_client.upload_blob(test_content, overwrite=True)
        
        # Verify and clean up
        blob_url = blob_client.url
        blob_client.delete_blob()
        
        return func.HttpResponse(
            json.dumps({
                'status': 'success',
                'message': 'Storage connectivity test passed',
                'test_blob_url': blob_url
            }),
            status_code=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        logging.error(f"Storage test failed: {e}")
        return func.HttpResponse(
            json.dumps({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }),
            status_code=500,
            mimetype='application/json'
        )

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
    """Background worker: generate climate analysis, upload results, and POST to callback_url."""
    #raise Exception("TESTING - If you see this in logs, function is running!")
    logging.info(f"[{task_id}] Background task started for: {demo_ID} | {name} -> {lat},{lon}")
    
    temp_dir_to_cleanup = None
    
    try:
        # Initialize climate service with CDS credentials
        climate_service = ClimateAnalysisService(
            cds_url=CDS_API_URL,
            cds_key=CDS_API_KEY
        )
        
        # Generate climate analysis
        logging.info(f"[{task_id}] Starting climate analysis...")
        results = climate_service.generate_climate_analysis(
            lat=lat,
            lon=lon,
            demo_id=demo_ID
        )

        if results.get('success'):
            # Get temp directory for cleanup later
            temp_dir_to_cleanup = results.get('temp_dir')
            
            # Get and validate visualization path
            visualization_path = results.get('visualization_path')
            blob_name = f"{demo_ID}_climate_analysis.png"
            
            logging.info(f"[{task_id}] Visualization path received: {visualization_path}")
            
            if not visualization_path:
                raise Exception("Climate service did not return visualization_path")
            
            if not os.path.exists(visualization_path):
                raise Exception(f"Visualization file not found: {visualization_path}")
            
            file_size = os.path.getsize(visualization_path)
            logging.info(f"[{task_id}] Visualization file size: {file_size:,} bytes")
            
            if file_size == 0:
                raise Exception("Visualization file is empty")
            
            # Attempt blob upload with detailed logging
            try:
                logging.info(f"[{task_id}] Attempting blob upload...")
                visualization_url = _upload_blob(visualization_path, blob_name)
                logging.info(f"[{task_id}] ✓ Blob upload successful: {visualization_url}")
                
                # Verify the URL is accessible (optional)
                if visualization_url.startswith('http'):
                    logging.info(f"[{task_id}] Blob URL generated: {visualization_url}")
                
            except Exception as upload_error:
                logging.error(f"[{task_id}] ✗ Blob upload failed: {upload_error}")
                # Don't use fallback file:// URLs in production
                raise Exception(f"Failed to upload visualization to blob storage: {upload_error}")

            payload = {
                'firstname': name,
                'demoID': demo_ID,
                'status': 'success',
                'coordinates': results.get('coordinates'),
                'scenarios_processed': results.get('scenarios_processed'),
                'year_range': results.get('year_range'),
                'climate_visualization': visualization_url
            }
        else:
            # Handle error case
            raise Exception(results.get('error', 'Unknown error in climate analysis'))

        logging.info(f"[{task_id}] Sending success payload to callback")
        
        # POST results to callback
        try:
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(callback_url, json=payload, headers=headers, timeout=30)
            logging.info(f"[{task_id}] Callback POST status: {resp.status_code}")
            
            if resp.status_code not in [200, 201, 202]:
                logging.warning(f"[{task_id}] Unexpected callback response: {resp.text[:200]}")
            
        except Exception as callback_error:
            logging.exception(f"[{task_id}] Callback POST failed: {callback_error}")

    except Exception as e:
        logging.exception(f"[{task_id}] Processing failed: {e}")

        # Notify callback about failure
        try:
            failure_payload = {
                'demoID': demo_ID,
                'status': 'failed',
                'error': str(e),
                'firstname': name
            }
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(callback_url, json=failure_payload, headers=headers, timeout=10)
            logging.info(f"[{task_id}] Failure notification sent, status: {resp.status_code}")
        except Exception as callback_error:
            logging.exception(f"[{task_id}] Failed to notify callback of failure: {callback_error}")
    finally:
        # Clean up temp directory
        if temp_dir_to_cleanup:
            try:
                import shutil
                shutil.rmtree(temp_dir_to_cleanup)
                logging.info(f"[{task_id}] Cleaned up temp directory: {temp_dir_to_cleanup}")
            except Exception as cleanup_error:
                logging.warning(f"[{task_id}] Failed to clean up temp directory: {cleanup_error}")
        
        logging.info(f"[{task_id}] Background task completed")


@app.route(route="ClimateHttpTrigger", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def climate(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for climate analysis requests."""
    logging.info('ClimateHttpTrigger: HTTP trigger received a request.')

    try:
        body = req.get_json()
    except ValueError:
        body = None

    lat = req.params.get('lat') or (body and body.get('lat'))
    lon = req.params.get('lon') or (body and body.get('lon'))
    name = req.params.get('firstname') or (body and body.get('firstname'))

    if lat is None or lon is None:
        logging.error(f'lat or lon is missing: {lat}, {lon}')
        return func.HttpResponse(
            json.dumps({'error': 'Please pass lat and lon in query string or JSON body'}),
            status_code=400,
            mimetype='application/json'
        )

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception as e:
        logging.error(f'invalid lat or lon: {lat}, {lon}')
        return func.HttpResponse(
            json.dumps({'error': 'lat and lon must be numbers', 'details': str(e)}),
            status_code=400,
            mimetype='application/json'
        )
    
    demo_ID = req.params.get('id') or (body and body.get('id'))
    if demo_ID is None:
        logging.error(f'demoid is missing: {demo_ID}')
        return func.HttpResponse(
            json.dumps({'error': 'the demo id is missing'}),
            status_code=400,
            mimetype='application/json'
        )
    
    # Validate CDS credentials
    if not CDS_API_URL or not CDS_API_KEY:
        logging.error('CDS API credentials not configured')
        return func.HttpResponse(
            json.dumps({'error': 'CDS API credentials not configured. Please set CDS_API_URL and CDS_API_KEY environment variables.'}),
            status_code=500,
            mimetype='application/json'
        )

    # Start background processing
    task_id = str(uuid.uuid4())
    logging.info(f'[{task_id}] About to start processing')
    
    # Option 1: Run synchronously (for testing)
    _process_and_callback(lat, lon, name, CALLBACK_URL, task_id, demo_ID)
    
    # Option 2: Run asynchronously (uncomment for production)
    # thread = threading.Thread(
    #     target=_process_and_callback,
    #     args=(lat, lon, name, CALLBACK_URL, task_id, demo_ID),
    #     daemon=True
    # )
    # thread.start()

    resp = {
        'task_id': task_id,
        'status': 'accepted',
        'message': 'Climate analysis started; results will be POSTed to callback_url'
    }
    
    return func.HttpResponse(json.dumps(resp), status_code=202, mimetype='application/json')


@app.route(route="HealthCheck", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint."""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'cds_configured': bool(CDS_API_URL and CDS_API_KEY),
        'storage_configured': bool(AZURE_STORAGE_CONNECTION_STRING),
        'storage_container': AZURE_STORAGE_CONTAINER_NAME,
        'callback_configured': bool(CALLBACK_URL)
    }
    
    return func.HttpResponse(
        json.dumps(health_status),
        status_code=200,
        mimetype='application/json'
    )