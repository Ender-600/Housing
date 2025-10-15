import os, httpx
from typing import List
from dotenv import load_dotenv

# Optional: Install googlemaps library for additional helper functions
# pip install googlemaps
try:
    import googlemaps as gmaps
except ImportError:
    gmaps = None

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
# Places API v1 Nearby Search (New): POST https://places.googleapis.com/v1/places:searchNearby
PLACES_ENDPOINT = "https://places.googleapis.com/v1/places:searchNearby"

async def count_places_nearby(lat: float, lon: float, *, included_types: List[str], radius_m: int = 1000) -> int:
    if not API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is missing")
    headers = {
        "X-Goog-Api-Key": API_KEY,
        # Request place ID and display name
        # Note: nextPageToken is a top-level field and is always returned
        "X-Goog-FieldMask": "places.id,places.displayName",
    }
    payload = {
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius_m}},
        "includedTypes": included_types,
    }
    total = 0
    async with httpx.AsyncClient(timeout=30) as client:
        token = None
        for _ in range(3):  # Nearby Search returns up to 3 pages
            if token:
                payload["pageToken"] = token
            resp = await client.post(PLACES_ENDPOINT, headers=headers, json=payload)
            if resp.status_code != 200:
                error_detail = resp.text
                raise RuntimeError(f"Google Places API error {resp.status_code}: {error_detail}")
            data = resp.json()
            places = data.get("places", [])
            total += len(places)
            # Print each place name
            for place in places:
                name = place.get("displayName", {}).get("text", "Unknown")
                print(f"  - {name}")
            token = data.get("nextPageToken")
            if not token:
                break
    return total



# ==== Tony's code ====
# Note: These functions require the 'googlemaps' library
# Install with: pip install googlemaps

# input is an adress and output is the corrdinates for that address if matches. 
def geocode(address: str):
    if gmaps is None:
        raise RuntimeError("googlemaps library is not installed. Install with: pip install googlemaps")
    client = gmaps.Client(key=API_KEY)
    g = client.geocode(address)
    if not g:
        return None
    loc = g[0]["geometry"]["location"]
    return loc["lat"], loc["lng"]  # could be ignored

def nearest_place(lat: float, lng: float, place_type: str = "school", radius_m: int = 5000):
    """Return (name, place_id, lat, lng) for the closest place of given type."""
    if gmaps is None:
        raise RuntimeError("googlemaps library is not installed. Install with: pip install googlemaps")
    client = gmaps.Client(key=API_KEY)
    r = client.places_nearby(location=(lat, lng), radius=radius_m, type=place_type, rank_by=None)
    if not r.get("results"):
        return None
    p = r["results"][0]
    loc = p["geometry"]["location"]
    return p["name"], p["place_id"], loc["lat"], loc["lng"]

def drive_minutes_between(orig, dest): # from a sepcific address to another
    if gmaps is None:
        raise RuntimeError("googlemaps library is not installed. Install with: pip install googlemaps")
    client = gmaps.Client(key=API_KEY)
    res = client.distance_matrix(origins=[orig], destinations=[dest], mode="driving", departure_time="now")
    el = res["rows"][0]["elements"][0]
    if el["status"] != "OK":
        return None
    seconds = el.get("duration_in_traffic", el["duration"])["value"]
    return round(seconds / 60, 1)

def time_to_nearest_type(address: str, place_type: str): # combine all three together
    geo = geocode(address)
    if not geo:
        return None, None
    lat, lng = geo
    np = nearest_place(lat, lng, place_type=place_type)
    if not np:
        return None, None
    name, place_id, plat, plng = np
    minutes = drive_minutes_between(address, f"place_id:{place_id}")
    return name, minutes




if __name__ == "__main__":
    import asyncio
    
    async def test_api():
        # Test coordinates: UIUC campus area
        lat, lon = 40.1020, -88.2272
        print(f"Testing Google Places API with coordinates: ({lat}, {lon})")
        
        try:
            # Test counting restaurants nearby
            print("\n1. Testing count_places_nearby for restaurants...")
            count = await count_places_nearby(lat, lon, included_types=["restaurant"], radius_m=1000)
            print(f"   ✓ Found {count} restaurants within 1km")
            
            # Test counting cafes nearby
            print("\n2. Testing count_places_nearby for cafes...")
            count = await count_places_nearby(lat, lon, included_types=["cafe"], radius_m=500)
            print(f"   ✓ Found {count} cafes within 500m")
            
            print("\n✅ Google Places API test completed successfully!")
        except Exception as e:
            print(f"\n❌ Google Places API test failed: {e}")
            raise
    
    asyncio.run(test_api())

    print(drive_minutes_between("818 Sedgegrass Dr, Champaign, IL 61822", "3204 Cypress Creek Rd, Champaign, IL 61822"))
    print(time_to_nearest_type("818 Sedgegrass Dr, Champaign, IL 61822", "hospital"))