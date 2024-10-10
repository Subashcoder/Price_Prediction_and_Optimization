import numpy as np
import pandas as pd
import os
import glob
from Event_data import Music_Event_data, Event_data


# Function to concat the files
def Merging_files(file_name):
    path_pattern = f'Data/*/{file_name}' 
    dataframes = []

    # Using glob to get a list of all matching file paths
    all_files = glob.glob(path_pattern)

    for f in all_files:
        month = os.path.basename(os.path.dirname(f))
        
        df = pd.read_csv(f)
        
        df['Month'] = month
        
        dataframes.append(df)
        
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# listing_merge = Merging_files('listings.csv.gz')
# print('Merge successfully')
# reviews_merge = Merging_files('reviews.csv.gz')
# print('Merge successfully')
# calendar_merge = Merging_files('calendar.csv.gz')
# print('Merge successfully')

# listing_merge.to_csv('./Data/Merge/listing.csv.gz', index=False)
# print('Saved successfully')
# reviews_merge.to_csv('./Data/Merge/reviews.csv.gz', index=False)
# print('Saved successfully')
# calendar_merge.to_csv('./Data/Merge/calendar.csv.gz', index=False)
# print('Saved successfully')

# function to change the str to pd_datetime
def datetime_conversion(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    return df


# event_data = Event_data('2023-09-00T21:32:45.107Z', '2024-01-00T21:32:45.107Z', 'toronto_event_oct_dec.csv')

# toronto_event_oct23_sep = pd.concat([pd.read_csv('./Data/toronto_events.csv'), pd.read_csv('./Data/toronto_event_oct_dec.csv')], ignore_index=True)

# toronto_event_oct23_sep.to_csv('./Data/Merge/Event_Merge.csv.gz')
# print('saved successfully')

calender = pd.read_csv('./Data/Merge/calendar.csv.gz', low_memory=False)
calender_cleaned = datetime_conversion(calender, 'date')
event = pd.read_csv('./Data/Merge/Event_Merge.csv.gz', low_memory=False)
event_cleaned = datetime_conversion(event, 'Date')

# Joining the Calendar listing data with events data
calender_event = calender_cleaned.merge(event_cleaned, 
                                        how = 'left', 
                                        left_on='date', 
                                        right_on='Date', 
                                        suffixes=('_cal', '_event'))

calender_event.to_csv('./Date/Merge/calender.csv.gz', index=False)
print('Done!!!')
                                                                                    

