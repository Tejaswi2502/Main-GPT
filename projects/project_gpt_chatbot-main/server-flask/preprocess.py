# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# import pandas as pd

# # Connect to Google Sheets
# scope = ['https://www.googleapis.com/auth/spreadsheets',
#          "https://www.googleapis.com/auth/drive"]

# credentials = ServiceAccountCredentials.from_json_keyfile_name("gs_credentials.json", scope)
# client = gspread.authorize(credentials)

#create a blank sheet
# sheet = client.create("gptDatabase")
# sheet.share('anshuman.parhi@stl.tech', perm_type='user', role='writer')

