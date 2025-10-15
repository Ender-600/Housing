# MTD API Test Suite

## Summary

Complete test suite for `code/services/mtd.py` `bus_stops_within_1km` API function.

### Files Created

```
Housing/
├── tests/
│   ├── __init__.py
│   ├── test_mtd.py       # 12 test cases
│   └── README.md          # Documentation
├── code/__init__.py       # Package init
├── pytest.ini             # Test configuration
├── requirements.txt       # Updated with test deps
└── venv/                  # Virtual environment
```

### Test Coverage (12 tests, 100% pass)

- ✅ Basic functionality (multiple stops, no stops, CSV data)
- ✅ Distance filtering and boundary conditions
- ✅ Empty response handling
- ✅ Batch CSV processing
- ✅ Parameter validation (max_to_fetch, API key)
- ✅ Error handling (HTTP errors, missing key)
- ✅ Integration with 117 CSV listings

### Tech Stack

- **pytest 8.4.2** - Test framework
- **pytest-asyncio** - Async test support
- **pytest-httpx** - HTTP mocking
- **pandas** - CSV data handling

### Usage

```bash
# Activate environment
source venv/bin/activate

# Run tests
pytest tests/test_mtd.py -v

# Result: 12 passed in 0.62s ✓
```

### Key Features

1. **Fully isolated** - No actual API calls
2. **Real data** - Uses `listings_summary.csv` coordinates
3. **Comprehensive** - Tests all functions and edge cases
4. **Async support** - Proper async function testing
5. **100% success rate** - All tests passing

### Data Source

- 117 Champaign housing listings with lat/lon
- Tests use first 10 records for batch testing
- Example: 40.08166, -88.255165 (6 Regent Ct)

---

**Date**: Oct 15, 2025  
**Tests**: 12/12 passed ✓  
**Python**: 3.13.3  
**Framework**: pytest 8.4.2
