import httpx
from shapely.geometry import Point, shape

# ArcGIS feature layer (police beats polygons)
ARCGIS_LAYER = "https://gisportal.champaignil.gov/ms/rest/services/Open_Data/Open_Data/MapServer/25/query"

async def police_beat_for_point(lat: float, lon: float):
    """
    Returns beat NAME that contains point, or None.
    """
    params = {
        "f": "geojson",
        "where": "1=1",
        "outFields": "NAME",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(ARCGIS_LAYER, params=params)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if feats:
            return feats[0]["properties"].get("NAME")
    return None

if __name__ == "__main__":
    import asyncio
    
    async def test_api():
        # Test coordinates: UIUC campus area
        lat, lon = 40.1020, -88.2272
        print(f"Testing Police Beats API with coordinates: ({lat}, {lon})")
        
        try:
            # Test police beat lookup
            print("\n1. Testing police_beat_for_point...")
            beat = await police_beat_for_point(lat, lon)
            if beat:
                print(f"   ✓ Police beat: {beat}")
            else:
                print(f"   ✗ No police beat found for this location")
            
            print("\n✅ Police Beats API test completed successfully!")
        except Exception as e:
            print(f"\n❌ Police Beats API test failed: {e}")
            raise
    
    asyncio.run(test_api())
