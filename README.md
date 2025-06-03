# Distance and Duration via Google Routes API

This script calculates travel distances and durations between multiple origins and destinations using the Google Routes API. It currently supports postcode-based origins, with the ability to expand to other origin formats in the future.

# How it works
- Converts postcodes to coordinates using Google Geocoding API
- Calculates driving distances and durations between origins and destinations, without traffic aware.
- Use cache to reduce API calls
- Batch processing. (RouteMatrix)


# Example .env
```
GOOGLE_MAPS_API_KEY=.......

ORIGIN_CSV=origins.csv
DESTINATION_CSV=destinations.csv
OUTPUT_CSV=Distance.csv

BATCH_SIZE=50  # defaults to 10 
# ori + desti < 50
# ori * desti < 625
```