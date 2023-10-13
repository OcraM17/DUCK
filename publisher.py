from __future__ import print_function
import os.path
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
from pygit2 import Repository
import time
from opts import OPT as opt

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Open the Google Sheet using credentials from the JSON key file
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = "1AdZdZJERaUironNl4vesg0e6k5Fk7BDEt-VNxXvgVSk"


import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def publish(data, range_name):
    """Publishes data to a Google Sheet.

    Args:
        data (list): The data to be published.
        range_name (str): The range in the Google Sheet where the data will be appended.

    Returns:
        None
    """
    credentials = None
    json_key_file_path = opt.root_folder + 'publisher_key.json'

    # Check if client_secrets.json file exists
    if os.path.exists('client_secrets.json'):
        credentials = Credentials.from_service_account_file(json_key_file_path, scopes=['https://www.googleapis.com/auth/spreadsheets'])

    try:
        # Build the Google Sheets API service
        service = build('sheets', 'v4', credentials=credentials)

        # Get the spreadsheets resource
        sheet = service.spreadsheets()

        # Prepare the values to be appended
        values = data

        # Get the last row in the range
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=range_name).execute()
        last_row = len(result.get('values', [])) + 1

        # Create the range to append the values
        range_to_append = range_name[:-4] + f'!A{last_row}:A{last_row}'

        # Create the request body
        body = {
            'values': values,
            'majorDimension': 'ROWS'
        }

        # Create the append request
        request = sheet.values().append(
            spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=range_to_append,
            valueInputOption='USER_ENTERED',
            body=body
        )

        # Execute the request
        response = request.execute()

        # Print success message
        print('Data written successfully.')

    except HttpError as err:
        print(err)

def push_results(args, df_or_model=None, df_un_model=None, df_rt_model=None):
    blacklist = ["__module__", "__dict__", "__weakref__", "__doc__", "cuda","device", "n_classes", "classes_per_exp", "details", "num_workers", "root_folder", "verboseMLP", "data_path"]
    
    repo = Repository(".")
    branch_name = repo.head.name.split("/")[-1]
    last_commit_id = repo[repo.head.target].hex
    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    params = [str(vars(args)[k]) for k in vars(args) if k not in blacklist] + [date_time, branch_name, last_commit_id]
    range_name = 'Params!A:P'
    
    publish_with_retries([params], range_name)
    
    range_name = 'Results!A:P'
    results = [str(vars(args)["run_name"])]
    n_df_params = 11
    if df_or_model is not None:
        results.extend([val for val in df_or_model.mean(axis=0)])
    else:
        results.extend([None]*n_df_params)
    if df_un_model is not None:
        results.extend([val for val in df_un_model.mean(axis=0)])
    else:
        results.extend([None]*(n_df_params+1))
    if df_rt_model is not None:
        results.extend([val for val in df_rt_model.mean(axis=0)])
    else:
        results.extend([None]*n_df_params)
    results = [str(r) for r in results]
    
    publish_with_retries([results], range_name)
    
#define publish with retries function to prevent timeot error
def publish_with_retries(data, range_name):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            publish(data, range_name)
            break
        except:
            print("Retrying...")
            time.sleep(5)
            retries += 1


#define main beahvior

if __name__ == "__main__":
    class ExampleClass:
        def __init__(self):
            self.attribute1 = 10
            self.attribute2 = "Hello"

    # Create an instance of ExampleClass
    obj = ExampleClass()

    push_results(obj)