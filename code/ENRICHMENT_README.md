# Housing Data Enrichment Tool

This tool adds new attributes from multiple API services to existing housing listing CSV data.

## Features

The tool automatically enriches each listing with the following information:

### 1. Census Data (Census API)

- `median_income`: Median household income for the block group

### 2. Public Transportation (MTD API)

- `bus_stops_1km`: Number of bus stops within 1km

### 3. Police Beat Information (Police Beats API)

- `police_beat`: Police beat name for the listing location

### 4. Nearby Amenities (Google Places API)

Configurable amenity types, defaults include:

- `restaurants_nearby`: Number of nearby restaurants
- `cafes_nearby`: Number of nearby cafes
- `schools_nearby`: Number of nearby schools
- `parks_nearby`: Number of nearby parks
- `gyms_nearby`: Number of nearby gyms
- `supermarkets_nearby`: Number of nearby supermarkets

### 5. Driving Time to Key Landmarks (Google Routes API)

Default landmarks:

- `drive_to_uiuc_main_quad_min`: Driving time to UIUC Main Quad (minutes)
- `drive_to_downtown_champaign_min`: Driving time to Downtown Champaign (minutes)
- `drive_to_carle_hospital_min`: Driving time to Carle Hospital (minutes)
- `drive_to_memorial_stadium_min`: Driving time to Memorial Stadium (minutes)
- `drive_to_willard_airport_min`: Driving time to Willard Airport (minutes)

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run data enrichment script
python code/enrich_listings.py \
  --input data/listings_details_allcities.csv \
  --output data/listings_enriched.csv
```

### Advanced Options

```bash
# Custom batch size (number of listings to process simultaneously)
python code/enrich_listings.py -i data/listings.csv -o data/enriched.csv --batch-size 10

# Start from specific index (to resume interrupted runs)
python code/enrich_listings.py -i data/listings.csv -o data/enriched.csv --start-idx 100

# Skip certain API calls (to save cost or time)
python code/enrich_listings.py -i data/listings.csv -o data/enriched.csv --skip-routes --skip-places

# Custom Google Places search radius (meters)
python code/enrich_listings.py -i data/listings.csv -o data/enriched.csv --places-radius 500
```

### Complete Parameter List

- `--input, -i`: Input CSV file path (default: `data/listings_details_allcities.csv`)
- `--output, -o`: Output CSV file path (default: `data/listings_enriched.csv`)
- `--batch-size, -b`: Number of listings to process in parallel (default: 5)
- `--start-idx`: Starting index for resuming interrupted runs (default: 0)
- `--skip-census`: Skip Census API calls
- `--skip-mtd`: Skip MTD API calls
- `--skip-police-beat`: Skip Police Beat API calls
- `--skip-places`: Skip Google Places API calls
- `--skip-routes`: Skip Google Routes API calls
- `--places-radius`: Google Places search radius in meters (default: 1000)

## Environment Configuration

Make sure your `.env` file contains all necessary API keys:

```bash
# Google APIs
GOOGLE_MAPS_API_KEY=your_google_api_key_here

# Census API
CENSUS_API_KEY=your_census_api_key_here

# MTD (CUMTD) API
CUMTD_API_KEY=your_mtd_api_key_here
```

## Important Notes

### API Costs

- **Google Places API**: $0.032 USD per Nearby Search call
- **Google Routes API**: ~$0.005-0.010 USD per Route Matrix call
- **Census API**: Free
- **MTD API**: Free
- **Police Beats API**: Free

### Rate Limiting

- The script uses batching and delays to avoid hitting API rate limits
- Default: processes 5 listings per batch with 1 second delay between batches
- Adjust parallelism with `--batch-size`

### Resuming Interrupted Runs

- The script automatically saves intermediate results to `*_intermediate.csv` after each batch
- If interrupted, use `--start-idx` to resume from the last position

### Data Quality

- Listings with missing coordinates will be skipped
- API errors will not interrupt the entire process, only logged and continue
- Each new column shows the percentage of non-null values

## Example Output

```
📂 Loading data from data/listings_details_allcities.csv...
✓ Loaded 150 listings

⚙️  Enrichment configuration:
  • Census API: ✓
  • MTD API: ✓
  • Police Beats API: ✓
  • Google Places API: ✓
  • Google Routes API: ✓
  • Places search radius: 1000m
  • Batch size: 5
  • Starting from index: 0

============================================================
Starting enrichment of 150 listings...
============================================================

--- Processing batch 1: listings 1 to 5 ---

[1/150] 1007 Rainbow Vw, Urbana, IL 61802
    ✓ Median income: $65,432
    ✓ Bus stops: 3
    ✓ Police beat: Beat 4
    ✓ Restaurants: 15
    ✓ Cafes: 5
    ✓ Schools: 2
    ...
    ✓ Drive to UIUC Main Quad: 8.5 min

  💾 Saved intermediate results to data/listings_enriched_intermediate.csv

...

============================================================
✅ Enrichment complete!
============================================================

💾 Saving enriched data to data/listings_enriched.csv...
✓ Saved 150 enriched listings

📊 Added 15 new columns:
  • median_income: 148/150 values (98.7%)
  • bus_stops_1km: 150/150 values (100.0%)
  • police_beat: 145/150 values (96.7%)
  • restaurants_nearby: 150/150 values (100.0%)
  • cafes_nearby: 150/150 values (100.0%)
  ...

✅ Done!
```

## Customization

To customize amenity types or landmarks:

1. **Nearby amenities**: Edit the `place_types` dictionary in `enrich_listings.py`
2. **Landmarks**: Edit the `LANDMARKS` dictionary in `utils/landmarks.py`

## Troubleshooting

### Issue: "GOOGLE_MAPS_API_KEY is missing"

**Solution**: Ensure `.env` file exists and contains the correct API key

### Issue: "API error 429: Rate limit exceeded"

**Solution**: Reduce `--batch-size` or increase `batch_delay_seconds` in configuration

### Issue: Script is running slowly

**Solutions**:

- Use `--skip-*` flags to skip unnecessary APIs
- Reduce `--places-radius`
- Limit the number of listings to process

### Issue: Out of memory

**Solution**: Reduce `--batch-size`

## Further Development

To add new API services:

1. Create a new service module in `code/services/`
2. Add new API calls in the `enrich_listing()` function
3. Add corresponding toggle options in the configuration
4. Update this README

## License

This tool is part of the Housing project.
