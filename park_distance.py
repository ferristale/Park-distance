import requests
import json
from typing import Dict, Optional, List, Set, Tuple
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pathlib

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
            
        # Initialize cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        pathlib.Path(self.cache_dir).mkdir(exist_ok=True)
            
        # Initialize caches
        self.coordinates_cache = {}
        self.coordinates_file = os.path.join(self.cache_dir, 'coordinates.csv')
        self.distance_cache = {}
        self.distance_cache_file = os.path.join(self.cache_dir, 'distance_cache.csv')
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
                logger.info(f"Loaded {len(self.distance_cache)} distances from cache")
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

    def get_google_distance_batch(self, origins: List[str], destinations: List[Dict]) -> Tuple[List[Dict], Set[str]]:
        """Calculate distances for a batch of origins and destinations using Google Routes API"""
        try:
            # Check cache first for all combinations
            results = []
            origins_to_process = []
            route_not_found_origins = set()
            
            for postcode in origins:
                found_in_cache = True
                for dest in destinations:
                    cache_key = f"{postcode}_{dest['place_id']}"
                    if cache_key in self.distance_cache:
                        results.append({
                            'Postcode': postcode,
                            'Destination': dest['name'],
                            'Distance': self.distance_cache[cache_key]['distance'],
                            'Travel_Time': self.distance_cache[cache_key]['duration']
                        })
                    else:
                        found_in_cache = False
                        break
                
                if not found_in_cache:
                    origins_to_process.append(postcode)
            
            if not origins_to_process:
                return results, route_not_found_origins
            
            # Prepare origins with coordinates
            origins_waypoints = []
            for postcode in origins_to_process:
                coordinates = self.get_coordinates_from_postcode(postcode)
                origins_waypoints.append({
                    "waypoint": {
                        "location": {
                            "latLng": {
                                "latitude": coordinates['lat'],
                                "longitude": coordinates['lng']
                            }
                        }
                    }
                })

            # Prepare destinations
            destinations_waypoints = [{
                "waypoint": {
                    "placeId": dest['place_id']
                }
            } for dest in destinations]

            base_url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
            request_body = {
                "origins": origins_waypoints,
                "destinations": destinations_waypoints,
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
            
            response = self.session.post(base_url, json=request_body, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                raise APIError("Invalid response format from Google Routes API")
            
            for element in data:
                origin_postcode = origins_to_process[element['originIndex']]
                
                if element.get('status', {}).get('code', 0) != 0 or element.get('condition') != 'ROUTE_EXISTS':
                    route_not_found_origins.add(origin_postcode)
                    continue
                
                if 'distanceMeters' not in element or 'duration' not in element:
                    route_not_found_origins.add(origin_postcode)
                    continue
                
                distance_km = round(element['distanceMeters'] / 1000, 2)
                duration_min = round(float(element['duration'][:-1]) / 60, 2)
                
                dest = destinations[element['destinationIndex']]
                
                # Cache the result
                cache_key = f"{origin_postcode}_{dest['place_id']}"
                self.distance_cache[cache_key] = {
                    'distance': distance_km,
                    'duration': duration_min
                }
                
                results.append({
                    'Postcode': origin_postcode,
                    'Destination': dest['name'],
                    'Distance': distance_km,
                    'Travel_Time': duration_min
                })
            
            self.save_distance_cache()
            return results, route_not_found_origins
            
        except Exception as e:
            logger.error(f"Error in Google Routes API batch request: {str(e)}")
            raise APIError(f"API error: {str(e)}")

    def process_csv(self, postcode_file: str, destination_file: str, output_file: str):
        """Process postcodes and destinations from separate files and calculate distances"""
        try:
            # Read postcodes file
            postcodes_df = pd.read_csv(postcode_file, na_values=['NA', 'nan', 'NaN', ''])
            if 'Postcode' not in postcodes_df.columns:
                raise ValueError("Postcode column not found in postcodes file")
            
            # Read destinations file
            destinations_df = pd.read_csv(destination_file, na_values=['NA', 'nan', 'NaN', ''])
            required_columns = ['destination', 'place_id']
            missing_columns = [col for col in required_columns if col not in destinations_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in destinations file: {', '.join(missing_columns)}")
            
            # Clean and process postcodes
            try:
                postcodes_df['Postcode'] = postcodes_df['Postcode'].fillna('').astype(str).str.strip()
                postcodes_df = postcodes_df[postcodes_df['Postcode'].str.len() > 0]
                postcodes_df['Postcode'] = postcodes_df['Postcode'].str.extract(r'(\d+)')[0]
                postcodes_df = postcodes_df.dropna(subset=['Postcode'])
                postcodes_df['Postcode'] = postcodes_df['Postcode'].str.zfill(4)
                postcodes_df = postcodes_df.drop_duplicates(subset=['Postcode'], keep='first')
                
                if postcodes_df.empty:
                    raise ValueError("No valid postcodes found in the input file")
                
                logger.info(f"Found {len(postcodes_df)} unique valid postcodes")
                
            except Exception as e:
                logger.error(f"Error cleaning postcodes: {str(e)}")
                raise ValueError(f"Failed to process postcodes: {str(e)}")
            
            # Process destinations
            destinations = []
            seen_place_ids = set()
            for _, row in destinations_df.iterrows():
                place_id = row['place_id']
                if place_id not in seen_place_ids:
                    destinations.append({
                        'name': row['destination'],
                        'place_id': place_id
                    })
                    seen_place_ids.add(place_id)
                else:
                    logger.warning(f"Skipping duplicate destination with place_id: {place_id}")
            
            if not destinations:
                raise ValueError("No valid destinations found in the input file")
            
            logger.info(f"Found {len(destinations)} unique destinations (removed {len(destinations_df) - len(destinations)} duplicates)")
            
            # Get batch size from environment variable or use default
            batch_size = int(os.getenv('BATCH_SIZE', '10'))
            logger.info(f"Using batch size of {batch_size}")
            
            # Process in batches
            all_results = []
            all_route_not_found_origins = set()
            
            for i in range(0, len(postcodes_df), batch_size):
                batch_postcodes = postcodes_df['Postcode'].iloc[i:i+batch_size].tolist()
                try:
                    batch_results, batch_errors = self.get_google_distance_batch(batch_postcodes, destinations)
                    all_results.extend(batch_results)
                    all_route_not_found_origins.update(batch_errors)
                    logger.info(f"Processed batch {i//batch_size + 1} of {(len(postcodes_df) + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                    # Add failed entries with None values
                    for postcode in batch_postcodes:
                        for dest in destinations:
                            all_results.append({
                                'Postcode': postcode,
                                'Destination': dest['name'],
                                'Distance': None,
                                'Travel_Time': None
                            })
            
            # Log all collected errors at the end
            if all_route_not_found_origins:
                logger.warning(f"No routes found for origins: {', '.join(sorted(all_route_not_found_origins))}")
            
            if not all_results:
                raise ValueError("No results were generated")
            
            pd.DataFrame(all_results).to_csv(output_file, index=False)
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

def main():
    try:
        calculator = ParkDistanceCalculator()
        calculator.process_csv(
            os.getenv('ORIGIN_CSV', 'origins.csv'),
            os.getenv('DESTINATION_CSV', 'destinations.csv'),
            os.getenv('OUTPUT_CSV', 'Distance.csv')
        )
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()