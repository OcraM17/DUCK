from __future__ import print_function
import os.path
import gspread
#from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials

#from google.auth.transport.requests import Request
#from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
from pygit2 import Repository
import time
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Open the Google Sheet using credentials from the JSON key file
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = "1AdZdZJERaUironNl4vesg0e6k5Fk7BDEt-VNxXvgVSk"


def publish(data, SAMPLE_RANGE_NAME):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    json_key_file_path = '/home/luigi/Work/MachineUnlearning/publisher_key.json'

    if os.path.exists('client_secrets.json'):
        # Specify the path to your JSON key file
        credentials = Credentials.from_service_account_file(json_key_file_path, scopes=['https://www.googleapis.com/auth/spreadsheets'])

        #credentials = Credentials.from_authorized_user_file(service_account_file, scopes=['https://www.googleapis.com/auth/spreadsheets'])

    try:
        # Create a service client for Google Sheets

        service = build('sheets', 'v4', credentials=credentials)

        sheet = service.spreadsheets()
        values = data # Modify this list with the data you want to write

        # Find the last written row in the specified range
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME).execute()
       
        last_row = len(result.get('values', [])) + 1

        # Append the data to the next empty row
        range_to_append = SAMPLE_RANGE_NAME[:-4] + f'!A{last_row}:A{last_row}'
        body = {
            'values': values,
            'majorDimension': 'ROWS'
        }
        request = sheet.values().append(
            spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=range_to_append,
            valueInputOption='USER_ENTERED',
            body=body
        )
        response = request.execute()
        print('Data written successfully.')

    except HttpError as err:
        print(err)

def push_results(args,  df_or_model=None, df_un_model=None, df_rt_model=None):
    #keys = [str(k) for k in vars(args)]#+["test accuracy", "task accuracy"]
    blacklist = ["__module__", "__dict__", "__weakref__", "__doc__", "cuda","device", "n_classes", "classes_per_exp", "details", "num_workers", "root_folder", "verboseMLP", "data_path"]
    repo = Repository(".")
    branch_name = repo.head.name.split("/")[-1]
    last_commit_id = repo[repo.head.target].hex
    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    params = [str(vars(args)[k]) for k in vars(args) if k not in blacklist] + [date_time, branch_name, last_commit_id]
    SAMPLE_RANGE_NAME = 'Params!A:P'
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            publish([params], SAMPLE_RANGE_NAME)
            break
        except:
            print("Retrying...")
            time.sleep(5)

    SAMPLE_RANGE_NAME = 'Results!A:P'
    results = [str(vars(args)["run_name"])]
    if df_or_model is not None:
        results.extend([df_or_model[i][0] for i in df_or_model])
    else:
        results.extend([None]*7)
    if df_un_model is not None:
        results.extend([df_un_model[i][0] for i in df_un_model])
    else:
        results.extend([None]*7)
    if df_rt_model is not None:
        results.extend([df_rt_model[i][0] for i in df_rt_model])
    else:
        results.extend([None]*7)
    results = [str(r) for r in results]

    #Code to prevent timeout error
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            publish([results], SAMPLE_RANGE_NAME)
            break
        except:
            print("Retrying...")
            time.sleep(5)
            retries += 1
 
    #publish([keys])
#define main beahvior
if __name__ == "__main__":
    class ExampleClass:
        def __init__(self):
            self.attribute1 = 10
            self.attribute2 = "Hello"

    # Create an instance of ExampleClass
    obj = ExampleClass()

    push_results(obj)