import requests
import json
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class ParkDistanceCalculator:
    def __init__(self):
        # Load environment variables
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            raise FileNotFoundError(f".env file not found at {env_path}")
            
        try:
            load_dotenv(env_path, override=True)
        except Exception as e:
            raise ValueError(f"Failed to load .env file: {str(e)}")
        
        # Initialize API key
        self.google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.google_api_key or not self.google_api_key.strip() or not self.google_api_key.startswith('AIza'):
            raise ValueError("Invalid Google Maps API key")
            
        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Load destinations from .env
        try:
            destinations_str = os.getenv('PARK_DESTINATIONS')
            if not destinations_str:
                raise ValueError("PARK_DESTINATIONS not found in .env file")
            
            self.destinations = []
            for dest in destinations_str.split(';'):
                if not dest.strip():
                    continue
                try:
                    name, place_id = dest.split(',')
                    self.destinations.append({
                        'name': name.strip(),
                        'place_id': place_id.strip()
                    })
                except ValueError:
                    logger.error(f"Invalid destination format: {dest}")
                    continue
            
            if not self.destinations:
                raise ValueError("No valid destinations found in .env file")
                
        except Exception as e:
            logger.error(f"Error parsing destination configurations: {str(e)}")
            raise ValueError(f"Invalid destination format in .env file: {str(e)}")
            
        # Initialize caches
        self.coordinates_cache = {}
        self.coordinates_file = 'coordinates.csv'
        self.distance_cache = {}
        self.distance_cache_file = 'distance_cache.csv'
        self.load_coordinates_cache()
        self.load_distance_cache()

    def load_coordinates_cache(self):
        """Load coordinates from CSV file into memory cache"""
        try:
            if os.path.exists(self.coordinates_file):
                df = pd.read_csv(self.coordinates_file)
                for _, row in df.iterrows():
                    self.coordinates_cache[str(row['postcode'])] = {
                        'lat': float(row['latitude']),
                        'lng': float(row['longitude'])
                    }
                logger.info(f"Loaded {len(self.coordinates_cache)} coordinates from cache")
        except Exception as e:
            logger.error(f"Error loading coordinates cache: {str(e)}")
            self.coordinates_cache = {}

    def save_coordinates_cache(self):
        """Save coordinates cache to CSV file"""
        try:
            data = [{'postcode': postcode, 'latitude': coords['lat'], 'longitude': coords['lng']} 
                   for postcode, coords in self.coordinates_cache.items()]
            pd.DataFrame(data).to_csv(self.coordinates_file, index=False)
        except Exception as e:
            logger.error(f"Error saving coordinates cache: {str(e)}")

    def load_distance_cache(self):
        """Load distance and duration cache from CSV file"""
        try:
            if os.path.exists(self.distance_cache_file):
                df = pd.read_csv(self.distance_cache_file)
                for _, row in df.iterrows():
                    cache_key = f"{row['postcode']}_{row['place_id']}"
                    self.distance_cache[cache_key] = {
                        'distance': float(row['distance']),
                        'duration': float(row['duration'])
                    }
        except Exception as e:
            logger.error(f"Error loading distance cache: {str(e)}")
            self.distance_cache = {}

    def save_distance_cache(self):
        """Save distance and duration cache to CSV file"""
        try:
            data = [{'postcode': key.split('_')[0], 'place_id': key.split('_')[1], 
                    'distance': values['distance'], 'duration': values['duration']} 
                   for key, values in self.distance_cache.items()]
            pd.DataFrame(data).to_csv(self.distance_cache_file, index=False)
        except Exception as e:
            logger.error(f"Error saving distance cache: {str(e)}")

    def get_coordinates_from_postcode(self, postcode: str) -> Dict[str, float]:
        """Convert postcode to coordinates using Google Geocoding API or cache"""
        if postcode in self.coordinates_cache:
            return self.coordinates_cache[postcode]
            
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': f"{postcode}, Australia",
                'key': self.google_api_key,
                'region': 'au'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'OK':
                raise APIError(f"Geocoding failed: {data['status']}")
                
            location = data['results'][0]['geometry']['location']
            coordinates = {'lat': location['lat'], 'lng': location['lng']}
            
            self.coordinates_cache[postcode] = coordinates
            self.save_coordinates_cache()
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error in geocoding: {str(e)}")
            raise APIError(f"Geocoding error: {str(e)}")

    def get_google_distance(self, origin: str, destination: str) -> Dict:
        """Calculate distance using Google Routes API"""
        cache_key = f"{origin}_{destination}"
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        try:
            coordinates = self.get_coordinates_from_postcode(origin)
            
            base_url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
            request_body = {
                "origins": [{
                    "waypoint": {
                        "location": {
                            "latLng": {
                                "latitude": coordinates['lat'],
                                "longitude": coordinates['lng']
                            }
                        }
                    }
                }],
                "destinations": [{
                    "waypoint": {
                        "placeId": destination
                    }
                }],
                "travelMode": "DRIVE",
                "languageCode": "en-US",
                "regionCode": "AU",
                "units": "METRIC"
            }
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.google_api_key,
                'X-Goog-FieldMask': 'originIndex,destinationIndex,status,condition,distanceMeters,duration'
            }
            
            response = self.session.post(base_url, json=request_body, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or not data:
                raise APIError("No results returned from Google Routes API")
            
            element = data[0]
            
            if element.get('status', {}).get('code', 0) != 0:
                raise APIError(f"Google Routes API Error: {element.get('status', {}).get('message', 'Unknown error')}")
            
            if element.get('condition') != 'ROUTE_EXISTS':
                raise APIError(f"Route condition error: {element.get('condition')}")
            
            if 'distanceMeters' not in element or 'duration' not in element:
                raise APIError("Missing distance or duration in response")
            
            distance_km = round(element['distanceMeters'] / 1000, 2)
            duration_min = round(float(element['duration'][:-1]) / 60, 2)
            
            result = {'distance': distance_km, 'duration': duration_min}
            self.distance_cache[cache_key] = result
            self.save_distance_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Google Routes API: {str(e)}")
            raise APIError(f"API error: {str(e)}")

    def process_csv(self, input_file: str, output_file: str, postcode_column: str):
        """Process a CSV file containing postcodes and calculate distances to all destinations"""
        try:
            # Read CSV and handle NA values during reading
            df = pd.read_csv(input_file, na_values=['NA', 'nan', 'NaN', ''])
            
            if postcode_column not in df.columns:
                raise ValueError(f"Column '{postcode_column}' not found in CSV file")
            
            # Clean and extract numbers from postcodes
            try:
                # Convert to string, handle NA values, and clean
                df[postcode_column] = df[postcode_column].fillna('').astype(str).str.strip()
                
                # Remove rows with empty or NA values
                df = df[df[postcode_column].str.len() > 0]
                
                # Extract numbers and ensure they are 4 digits
                df[postcode_column] = df[postcode_column].str.extract(r'(\d+)')[0]
                df = df.dropna(subset=[postcode_column])  # Remove rows where no numbers were found
                df[postcode_column] = df[postcode_column].str.zfill(4)
                
                # Remove duplicates while preserving order
                df = df.drop_duplicates(subset=[postcode_column], keep='first')
                
                if df.empty:
                    raise ValueError("No valid numbers found in the input file")
                
                logger.info(f"Found {len(df)} unique valid postcodes")
                
            except Exception as e:
                logger.error(f"Error cleaning postcodes: {str(e)}")
                raise ValueError(f"Failed to process postcodes: {str(e)}")
            
            # Process unique postcodes
            results = []
            for postcode in df[postcode_column].unique():
                for dest in self.destinations:
                    try:
                        distance_result = self.get_google_distance(postcode, dest['place_id'])
                        results.append({
                            'Postcode': postcode,
                            'Park': dest['name'],
                            'Distance': distance_result['distance'],
                            'Travel_Time': distance_result['duration']
                        })
                    except Exception as e:
                        logger.error(f"Error processing postcode {postcode} for destination {dest['name']}: {str(e)}")
                        results.append({
                            'Postcode': postcode,
                            'Park': dest['name'],
                            'Distance': None,
                            'Travel_Time': None
                        })
            
            if not results:
                raise ValueError("No results were generated")
            
            pd.DataFrame(results).to_csv(output_file, index=False)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

def main():
    try:
        calculator = ParkDistanceCalculator()
        calculator.process_csv(
            os.getenv('INPUT_CSV', 'Postcode.csv'),
            os.getenv('OUTPUT_CSV', 'Distance.csv'),
            os.getenv('POSTCODE_COLUMN', 'Postcode')
        )
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main() 