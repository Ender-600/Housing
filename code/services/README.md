# Services API Testing

Each service file includes a built-in test function that can be run directly to verify the API is working properly.

## Running Tests

### 1. Census API Test

```bash
python code/services/census.py
```

Tests:

- Retrieve Census block group information from coordinates
- Query median household income for the area

### 2. Google Places API Test

```bash
python code/services/google_places.py
```

Tests:

- Count restaurants within 1km radius
- Count cafes within 500m radius

**Requires environment variable**: `GOOGLE_MAPS_API_KEY`

### 3. Google Routes API Test

```bash
python code/services/google_routes.py
```

Tests:

- Calculate drive time from UIUC campus to downtown
- Calculate transit time (if available)

**Requires environment variable**: `GOOGLE_MAPS_API_KEY`

### 4. MTD (CUMTD) API Test

```bash
python code/services/mtd.py
```

Tests:

- Count bus stops within 1km radius

**Requires environment variable**: `CUMTD_API_KEY`

### 5. Police Beats API Test

```bash
python code/services/police_beats.py
```

Tests:

- Find police beat for given coordinates

**No API key required** - Uses public ArcGIS data

## Test Coordinates

All tests use the same test coordinates:

- **Location**: UIUC campus area
- **Coordinates**: (40.1020, -88.2272)

## Environment Variables Setup

Before running tests, set up your API keys using a `.env` file:

1. **Copy the example file:**

   ```bash
   cp env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   CENSUS_API_KEY=optional_small_usage_without_key_ok
   GOOGLE_MAPS_API_KEY=your_actual_google_maps_key
   CUMTD_API_KEY=your_actual_cumtd_key
   ```

**Note**: All service files automatically load environment variables from `.env` using `python-dotenv`.

Alternatively, you can set environment variables directly in your shell:

```bash
export CENSUS_API_KEY="your_census_key"  # Optional for Census API
export GOOGLE_MAPS_API_KEY="your_google_key"
export CUMTD_API_KEY="your_mtd_key"
```

## Run All Tests at Once

You can create a simple script to run all tests:

```bash
for service in census google_places google_routes mtd police_beats; do
    echo "================================"
    echo "Testing: $service"
    echo "================================"
    python code/services/${service}.py
    echo ""
done
```
