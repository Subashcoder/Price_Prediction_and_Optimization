import requests
import os
from datetime import datetime, timedelta
import pandas as pd


#--------------- Toronto Open source data---------------------------

#formate
# startDateTime = 2024-10-00T21:32:45.107Z
# endDateTime = 2024-10-00T21:32:45.107Z
def Event_data(filename):
    starttime = f"{(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}T21:32:45.107Z"
    endtime = f"{datetime.now().strftime('%Y-%m-%d')}T21:32:45.107Z"

    url = f'https://secure.toronto.ca/cc_sr_v1/data/edc_eventcal_APR?limit=500&{starttime}&{endtime}'

    response = requests.get(url=url).json()

    print(len(response))

    # eventName = []
    # Date = []
    # event_long = []
    # event_lat = []
    # event_address = []
    # category = []


    # for data in response:
    #     calevent = data['calEvent']
    #     eventName.append(calevent['eventName'])
    #     Date.append([date['startDateTime'] for date in calevent['dates']])
    #     category.append(calevent['categoryString'])
    #     # long_lat = [value for key, value in enumerate(calevent['locations'][0]['coords'])]
    #     long_lat = dict(calevent['locations'][0]['coords'])
    #     try:
    #         long_lat = calevent['locations'][0].get('coords', {})
    #         event_long.append(long_lat.get('lng', 0))
    #         event_lat.append(long_lat.get('lat', 0))
    #         event_address.append(calevent['locations'][0]['address'])
    #     except (KeyError, IndexError, AttributeError):
    #         event_long.append(0)
    #         event_lat.append(0)
    #         event_address.append(None)
            
    # print(len(event_address), len(eventName), len(event_lat), len(event_long), len(Date))
    # df = pd.DataFrame({
    #                 'EventName': eventName,
    #                 'Date': Date,
    #                 'event_long': event_long,
    #                 'event_lat': event_lat,
    #                 'address': event_address,
    #                 'categoryString': category
    #                 })

    # print(df.head())
    # df_new = df.explode(column='Date')
    # print(df_new.head())
    # print(df_new.shape)

    # df_new.to_csv(filename, index=False)
    # print('Saved Successfully!!!')