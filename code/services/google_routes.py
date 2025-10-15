import os, httpx
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
ROUTE_MATRIX_ENDPOINT = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

async def drive_time_minutes(origin: Tuple[float, float], dest: Tuple[float, float], *, mode: str="DRIVE") -> Optional[float]:
    """
    Returns travel time in minutes using Google Routes API computeRouteMatrix.
    Requires billing-enabled project.
    """
    if not API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is missing")
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "originIndex,destinationIndex,duration,condition",
    }
    body = {
        "origins": [{"waypoint": {"location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}}}],
        "destinations": [{"waypoint": {"location": {"latLng": {"latitude": dest[0], "longitude": dest[1]}}}}],
        "travelMode": mode,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(ROUTE_MATRIX_ENDPOINT, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        # First element corresponds to (0,0)
        el = data[0]
        secs = parse_duration(el.get("duration", ""))
        return secs/60 if secs is not None else None

def parse_duration(google_duration: str):
    # Google duration is an RFC3339 duration like "123s"
    if not google_duration or not google_duration.endswith("s"):
        return None
    try:
        return float(google_duration[:-1])
    except ValueError:
        return None

if __name__ == "__main__":
    import asyncio
    
    async def test_api():
        # Test coordinates: UIUC campus to downtown Champaign
        origin = (40.1020, -88.2272)  # UIUC campus
        dest = (40.1164, -88.2434)     # Downtown Champaign
        print(f"Testing Google Routes API")
        print(f"Origin: {origin}")
        print(f"Destination: {dest}")
        
        try:
            # Test drive time
            print("\n1. Testing drive_time_minutes...")
            minutes = await drive_time_minutes(origin, dest, mode="DRIVE")
            if minutes:
                print(f"   ✓ Drive time: {minutes:.1f} minutes")
            else:
                print(f"   ✗ No route data available")
            
            # Test transit time (optional)
            print("\n2. Testing drive_time_minutes with TRANSIT mode...")
            minutes = await drive_time_minutes(origin, dest, mode="TRANSIT")
            if minutes:
                print(f"   ✓ Transit time: {minutes:.1f} minutes")
            else:
                print(f"   ⚠ Transit data not available")
            
            print("\n✅ Google Routes API test completed successfully!")
        except Exception as e:
            print(f"\n❌ Google Routes API test failed: {e}")
            raise
    
    asyncio.run(test_api())
