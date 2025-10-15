import os, httpx
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

CENSUS_KEY = os.getenv("CENSUS_API_KEY")

GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"

async def blockgroup_from_latlon(lat: float, lon: float):
    params = {
        "x": lon, "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(GEOCODER, params=params)
        r.raise_for_status()
        data = r.json()
        try:
            # Try to get Census Block Groups first (preferred)
            geos = data["result"]["geographies"].get("Census Block Groups", [])
            if not geos:
                # Fall back to 2020 Census Blocks which also contains block group info
                geos = data["result"]["geographies"]["2020 Census Blocks"]
            
            geo = geos[0]
            return {
                "state": geo["STATE"],
                "county": geo["COUNTY"],
                "tract": geo["TRACT"],
                "block_group": geo["BLKGRP"],
            }
        except (KeyError, IndexError) as e:
            # If the expected structure is not found, provide helpful error
            import json
            raise RuntimeError(f"Unexpected Census API response structure: {json.dumps(data, indent=2)}") from e

async def median_income_bg(state: str, county: str, tract: str, block_group: str, year: int = 2023):
    # Variable B19013_001E = Median household income (inflation-adjusted dollars)
    url = ACS_BASE.format(year=year)
    params = {
        "get": "NAME,B19013_001E",
        "for": f"block group:{block_group}",
        "in": f"state:{state} county:{county} tract:{tract}",
    }
    if CENSUS_KEY and CENSUS_KEY.lower() != "optional_small_usage_without_key_ok":
        params["key"] = CENSUS_KEY
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        rows = r.json()
        # rows[0] header, rows[1] value
        try:
            return int(rows[1][1]) if rows and len(rows) > 1 else None
        except Exception:
            return None

async def median_income_from_latlon(lat: float, lon: float, year: int = 2023):
    geo = await blockgroup_from_latlon(lat, lon)
    return await median_income_bg(geo["state"], geo["county"], geo["tract"], geo["block_group"], year=year)

if __name__ == "__main__":
    import asyncio
    
    async def test_api():
        # Test coordinates: UIUC campus area
        lat, lon = 40.1020, -88.2272
        print(f"Testing Census API with coordinates: ({lat}, {lon})")
        
        try:
            # Test blockgroup lookup
            print("\n1. Testing blockgroup_from_latlon...")
            geo = await blockgroup_from_latlon(lat, lon)
            print(f"   ✓ Block group: State={geo['state']}, County={geo['county']}, Tract={geo['tract']}, BG={geo['block_group']}")
            
            # Test median income
            print("\n2. Testing median_income_from_latlon...")
            income = await median_income_from_latlon(lat, lon)
            print(f"   ✓ Median household income: ${income:,}" if income else "   ✗ No income data available")
            
            print("\n✅ Census API test completed successfully!")
        except Exception as e:
            print(f"\n❌ Census API test failed: {e}")
            raise
    
    asyncio.run(test_api())
