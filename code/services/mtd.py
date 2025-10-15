import os, httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("CUMTD_API_KEY")
ENDPOINT = "https://developer.mtd.org/api/v2.2/json/getstopsbylatlon"

FT_PER_KM = 3280.8399

async def bus_stops_within_1km(lat: float, lon: float, *, max_to_fetch: int = 200) -> Optional[int]:
    if not API_KEY:
        raise RuntimeError("CUMTD_API_KEY is missing")
    params = {
        "key": API_KEY,
        "lat": lat,
        "lon": lon,
        "count": max_to_fetch,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(ENDPOINT, params=params)
        r.raise_for_status()
        data = r.json()
        stops = data.get("stops", []) or data.get("stop", []) or []
        # Filter by distance (feet)
        return sum(1 for s in stops if s.get("distance", FT_PER_KM * 2) <= FT_PER_KM)

if __name__ == "__main__":
    import asyncio
    
    async def test_api():
        # Test coordinates: UIUC campus area
        lat, lon = 40.1020, -88.2272
        print(f"Testing MTD (CUMTD) API with coordinates: ({lat}, {lon})")
        
        try:
            # Test bus stops count
            print("\n1. Testing bus_stops_within_1km...")
            count = await bus_stops_within_1km(lat, lon)
            if count is not None:
                print(f"   ✓ Found {count} bus stops within 1km")
            else:
                print(f"   ✗ No bus stop data available")
            
            print("\n✅ MTD API test completed successfully!")
        except Exception as e:
            print(f"\n❌ MTD API test failed: {e}")
            raise
    
    asyncio.run(test_api())
