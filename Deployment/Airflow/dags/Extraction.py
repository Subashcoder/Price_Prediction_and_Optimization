import os
import requests
from datetime import datetime, timedelta

def download_airbnb_data(base_url, destination_folder, year, month, **kwargs):
    """
    Download Airbnb listings data for a specific year and month by finding the correct day.
    Args:
        base_url (str): Base URL of the Airbnb data site.
        destination_folder (str): Local folder to save the downloaded files.
        year (int): Year of the data.
        month (int): Month of the data.
    """
    days_in_month = range(1, 32)  # Days 1 to 31
    found_file = False

    for day in days_in_month:
        date_string = f"{year}-{month:02d}-{day:02d}"
        data_string_for_filename = f'{year}-{month:02d}'
        url = f"{base_url}/{date_string}/data/listings.csv.gz"
        
        try:
            print(f"Trying {url} ...")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                
                # Filepath for saving data
                filename = f"listings_{data_string_for_filename}.csv.gz"
                file_path = os.path.join(destination_folder, filename)
                
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded and saved to {file_path}")
                found_file = True
                break
            
            elif response.status_code == 404:
                continue

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to {url}: {e}")
    
    if not found_file:
        print(f"No file found for {year}-{month:02d}.")

def download_last_12_months(**kwargs):
    """
    Downloads Airbnb data for the last 12 months dynamically.
    """
    base_url = "https://data.insideairbnb.com/canada/on/toronto"
    destination_folder = "/opt/Data"
    
    current_date = datetime.now()
    
    # Loop for the last 12 months
    for i in range(2):
        target_date = current_date - timedelta(days=30 * i)
        year = target_date.year
        month = target_date.month

        download_airbnb_data(base_url, destination_folder, year, month)
