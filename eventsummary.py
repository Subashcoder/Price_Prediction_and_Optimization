import pandas as pd

# Load the compressed dataset
input_file = "Data/merge/Event_Merge.csv.gz"  # Your gzipped input file path
data = pd.read_csv(input_file, compression='gzip')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Group by 'EventName' and calculate start and end date for each event
event_summary = data.groupby('EventName').agg(
    start_date=('Date', 'min'),
    end_date=('Date', 'max'),
    event_long=('event_long', 'first'),
    event_lat=('event_lat', 'first'),
    address=('address', 'first'),
    categoryString=('categoryString', 'first')
).reset_index()

# Format start and end dates to 'YYYY-MM-DD'
event_summary['start_date'] = event_summary['start_date'].dt.strftime('%Y-%m-%d')
event_summary['end_date'] = event_summary['end_date'].dt.strftime('%Y-%m-%d')

# Save the event summary to a new compressed CSV file
output_file = "Data/merge/Event_Summary.csv.gz"  # Specify your desired output file path
event_summary.to_csv(output_file, index=False, compression='gzip')

print(f"Event summary saved to {output_file}")
