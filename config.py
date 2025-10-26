from helper import SingletonMeta
import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

'''
Environment Variables:
Ensure that your production environment has the necessary environment variables set for
AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID, which are required for DefaultAzureCredential.
'''

class Config(metaclass=SingletonMeta):
  __vars = {}

  def __init__(self):
    # Replace with your Key Vault name and secret name
    key_vault_name = os.getenv('KEY_VAULT_NAME')
    KVUri = f"https://{key_vault_name}.vault.azure.net"

    # Initialize the client with DefaultAzureCredential
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)    # Store Key Vault client for CDS API credentials
    self.key_vault_client = client
    
    # Keep essential storage configuration for output files
    self.__vars['AZURE_STORAGE_CONNECTION_STRING'] = client.get_secret('AZURE-STORAGE-CONNECTION-STRING').value
    self.__vars['AZURE_STORAGE_CONTAINER_NAME'] = 'fzcairbus'
    
    # Output folder for generated images
    self.__vars['IMAGES_FOLDER'] = './tmp/images'

  def get (self, key: str) -> any:
    return self.__vars.get(key)
  
  def get_string (self, key: str) -> str:
    return str(self.get(key))

  def get_cds_url(self):
      """Retrieve CDS API URL from Key Vault"""
      try:
          return self.key_vault_client.get_secret("cds-api-url").value
      except Exception as e:
          print(f"Error retrieving CDS URL: {e}")
          return "https://cds.climate.copernicus.eu/api/v2"  # Default CDS URL
  
  def get_cds_api_key(self):
      """Retrieve CDS API key from Key Vault"""
      try:
          return self.key_vault_client.get_secret("cds-api-key").value
      except Exception as e:
          print(f"Error retrieving CDS API key: {e}")
          return None