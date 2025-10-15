# MTD API Tests

Tests for `code/services/mtd.py` - MTD (Champaign-Urbana Mass Transit District) API integration.

data/listings_summary.csv## Test Coverage

12 test cases covering:

- **Basic functionality** - Multiple stops, no stops, real CSV data
- **Distance filtering** - Boundary conditions, missing distance fields
- **Data formats** - Empty responses, various JSON structures
- **Parameters** - `max_to_fetch`, API request validation
- **Error handling** - Missing API key, HTTP errors
- **Integration** - Batch processing with CSV data

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/test_mtd.py -v

# Run specific test
pytest tests/test_mtd.py::TestBusStopsWithin1km::test_with_real_csv_data -v
```

## Test Data

- **Mock data**: Simulated API responses for unit tests
- **CSV data**: Real coordinates from `data/listings_summary.csv` (117 listings)

## Key Features

- ✅ No actual API calls (uses `pytest-httpx` mocks)
- ✅ No real API key needed (test key auto-set)
- ✅ Tests async functions with `pytest-asyncio`
- ✅ Uses real housing data from CSV

## Dependencies

```
pytest>=8.0
pytest-asyncio>=0.23
pytest-httpx>=0.30
pandas>=2.1
httpx>=0.27
```

## Test Options

```bash
# Verbose output
pytest tests/test_mtd.py -v

# Show failures only
pytest tests/test_mtd.py -x

# With coverage
pytest tests/test_mtd.py --cov=code/services/mtd
```

## Results

```
12 passed in 0.62s ✓
```
