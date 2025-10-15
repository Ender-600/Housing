import pytest
import pandas as pd
import sys
import os
from pathlib import Path
from httpx import Response

# Set test API key before importing module
os.environ.setdefault("CUMTD_API_KEY", "test_api_key_for_testing")

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import mtd module using importlib to avoid conflict with stdlib 'code' module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mtd_service",
    project_root / "code" / "services" / "mtd.py"
)
mtd_service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtd_service)

bus_stops_within_1km = mtd_service.bus_stops_within_1km
ENDPOINT = mtd_service.ENDPOINT
FT_PER_KM = mtd_service.FT_PER_KM


@pytest.fixture
def sample_listings():
    """Load sample listings data"""
    csv_path = Path(__file__).parent.parent / "data" / "listings_summary.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["latitude", "longitude"])
    return df


@pytest.fixture
def mock_api_response_multiple_stops():
    """Mock API response with multiple bus stops"""
    return {
        "stops": [
            {"stop_id": "1", "stop_name": "Stop 1", "distance": 500.0},
            {"stop_id": "2", "stop_name": "Stop 2", "distance": 1000.0},
            {"stop_id": "3", "stop_name": "Stop 3", "distance": 2500.0},
            {"stop_id": "4", "stop_name": "Stop 4", "distance": 3000.0},
            {"stop_id": "5", "stop_name": "Stop 5", "distance": 3280.0},  # ~1km
            {"stop_id": "6", "stop_name": "Stop 6", "distance": 3300.0},  # >1km
            {"stop_id": "7", "stop_name": "Stop 7", "distance": 5000.0},  # >1km
        ]
    }


@pytest.fixture
def mock_api_response_no_stops():
    """Mock API response with no stops"""
    return {"stops": []}


@pytest.fixture
def mock_api_response_empty_variants():
    """Different empty response formats"""
    return [
        {"stops": []},
        {"stops": None},
        {"stop": []},
        {"stop": None},
        {},
    ]


class TestBusStopsWithin1km:
    """Tests for bus_stops_within_1km function"""

    @pytest.mark.asyncio
    async def test_multiple_stops_within_1km(
        self, httpx_mock, mock_api_response_multiple_stops
    ):
        """Test with multiple stops within 1km"""
        httpx_mock.add_response(json=mock_api_response_multiple_stops)
        result = await bus_stops_within_1km(40.08166, -88.255165)
        assert result == 5  # Only stops <= FT_PER_KM

    @pytest.mark.asyncio
    async def test_no_stops(self, httpx_mock, mock_api_response_no_stops):
        """Test with no stops"""
        httpx_mock.add_response(json=mock_api_response_no_stops)
        result = await bus_stops_within_1km(40.08166, -88.255165)
        assert result == 0

    @pytest.mark.asyncio
    async def test_with_real_csv_data(self, httpx_mock, sample_listings):
        """Test with real lat/lon from CSV"""
        first_listing = sample_listings.iloc[0]
        lat = first_listing["latitude"]
        lon = first_listing["longitude"]

        mock_response = {
            "stops": [
                {"stop_id": "1", "distance": 1000.0},
                {"stop_id": "2", "distance": 2000.0},
                {"stop_id": "3", "distance": 3000.0},
            ]
        }

        httpx_mock.add_response(json=mock_response)
        result = await bus_stops_within_1km(lat, lon)
        assert result == 3
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_batch_csv_locations(self, httpx_mock, sample_listings):
        """Test batch processing of CSV locations"""
        for idx, row in sample_listings.head(5).iterrows():
            num_stops = idx % 4
            mock_response = {
                "stops": [
                    {"stop_id": str(i), "distance": 1000.0 * (i + 1)}
                    for i in range(num_stops)
                ]
            }
            httpx_mock.add_response(json=mock_response)
            result = await bus_stops_within_1km(row["latitude"], row["longitude"])
            assert result == num_stops
            assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_filter_by_distance(self, httpx_mock):
        """Test distance filtering logic"""
        mock_response = {
            "stops": [
                {"stop_id": "1", "distance": FT_PER_KM - 1},  # Should count
                {"stop_id": "2", "distance": FT_PER_KM},  # Should count
                {"stop_id": "3", "distance": FT_PER_KM + 1},  # Should not count
                {"stop_id": "4", "distance": FT_PER_KM * 2},  # Should not count
            ]
        }
        httpx_mock.add_response(json=mock_response)
        result = await bus_stops_within_1km(40.08166, -88.255165)
        assert result == 2

    @pytest.mark.asyncio
    async def test_stops_without_distance_field(self, httpx_mock):
        """Test stops with missing distance field"""
        mock_response = {
            "stops": [
                {"stop_id": "1", "distance": 1000.0},  # Has distance
                {"stop_id": "2"},  # Missing distance
                {"stop_id": "3", "distance": 2000.0},  # Has distance
            ]
        }
        httpx_mock.add_response(json=mock_response)
        result = await bus_stops_within_1km(40.08166, -88.255165)
        assert result == 2

    @pytest.mark.asyncio
    async def test_empty_response_variants(self, httpx_mock, mock_api_response_empty_variants):
        """Test various empty response formats"""
        for empty_response in mock_api_response_empty_variants:
            httpx_mock.add_response(json=empty_response)
            result = await bus_stops_within_1km(40.08166, -88.255165)
            assert result == 0

    @pytest.mark.asyncio
    async def test_max_to_fetch_parameter(self, httpx_mock):
        """Test max_to_fetch parameter"""
        httpx_mock.add_response(json={"stops": []})
        await bus_stops_within_1km(40.08166, -88.255165, max_to_fetch=100)
        request = httpx_mock.get_request()
        assert request is not None
        assert "count=100" in str(request.url)

    @pytest.mark.asyncio
    async def test_api_key_missing(self, monkeypatch):
        """Test missing API key"""
        monkeypatch.delenv("CUMTD_API_KEY", raising=False)

        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "mtd_reload",
            project_root / "code" / "services" / "mtd.py"
        )
        mtd_reload = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mtd_reload)

        with pytest.raises(RuntimeError, match="CUMTD_API_KEY is missing"):
            await mtd_reload.bus_stops_within_1km(40.08166, -88.255165)

    @pytest.mark.asyncio
    async def test_http_error_handling(self, httpx_mock):
        """Test HTTP error handling"""
        httpx_mock.add_response(status_code=500)
        with pytest.raises(Exception):
            await bus_stops_within_1km(40.08166, -88.255165)

    @pytest.mark.asyncio
    async def test_request_parameters(self, httpx_mock):
        """Test API request parameters"""
        httpx_mock.add_response(json={"stops": []})
        lat, lon = 40.08166, -88.255165
        await bus_stops_within_1km(lat, lon, max_to_fetch=150)

        request = httpx_mock.get_request()
        assert request is not None
        url_str = str(request.url)
        assert f"lat={lat}" in url_str
        assert f"lon={lon}" in url_str
        assert "count=150" in url_str
        assert "key=" in url_str


@pytest.mark.asyncio
async def test_integration_with_csv_sample(httpx_mock, sample_listings):
    """Integration test with CSV sample data"""
    results = []

    for idx, row in sample_listings.head(10).iterrows():
        num_stops = (idx * 3) % 8
        mock_response = {
            "stops": [
                {"stop_id": str(i), "distance": 500.0 * (i + 1)}
                for i in range(num_stops)
            ]
        }
        httpx_mock.add_response(json=mock_response)
        count = await bus_stops_within_1km(row["latitude"], row["longitude"])
        results.append(
            {
                "address": row["address"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "bus_stops_within_1km": count,
            }
        )

    assert len(results) == 10
    for result in results:
        assert "address" in result
        assert "bus_stops_within_1km" in result
        assert isinstance(result["bus_stops_within_1km"], int)
        assert result["bus_stops_within_1km"] >= 0

