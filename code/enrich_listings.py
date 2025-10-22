#!/usr/bin/env python3
"""
Data enrichment script to add API-derived attributes to housing listings CSV.

This script reads housing listings from CSV, enriches each listing with data from:
- Census API: median household income for the block group
- MTD API: number of bus stops within 1km
- Police Beats API: police beat name
- Google Places API: count of nearby amenities (restaurants, cafes, schools, etc.)
- Google Routes API: travel time to key landmarks

Quick Start (uses default settings):
    python code/enrich_listings.py

Usage with custom options:
    python code/enrich_listings.py --input data/listings_details_allcities.csv --output data/listings_enriched.csv
    
    # Skip expensive Google APIs
    python code/enrich_listings.py --skip-places --skip-routes
    
    # Custom batch size and radius
    python code/enrich_listings.py --batch-size 10 --places-radius 500
"""

import argparse
import asyncio
import pandas as pd
from typing import Dict, Optional, Any
import sys
from pathlib import Path

# Import service modules
from services.census import median_income_from_latlon
from services.mtd import bus_stops_within_1km
from services.police_beats import police_beat_for_point
from services.google_places import count_places_nearby
from services.google_routes import drive_time_minutes

# Import landmarks for calculating travel times
from utils.landmarks import LANDMARKS


async def enrich_listing(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a single listing with additional attributes from various APIs.
    
    Args:
        row: A pandas Series representing one listing
        config: Configuration dict with options like which APIs to call
    
    Returns:
        Dictionary of new attributes to add to the listing
    """
    lat = row['latitude']
    lon = row['longitude']
    
    enriched_data = {}
    
    # Skip if coordinates are missing
    if pd.isna(lat) or pd.isna(lon):
        print(f"  ⚠ Skipping {row.get('address', 'unknown')}: missing coordinates")
        return enriched_data
    
    try:
        # 1. Census data - Median household income
        if config.get('include_census', True):
            try:
                median_income = await median_income_from_latlon(lat, lon)
                enriched_data['median_income'] = median_income
                print(f"    ✓ Median income: ${median_income:,}" if median_income else "    - No income data")
            except Exception as e:
                print(f"    ✗ Census API error: {e}")
                enriched_data['median_income'] = None
        
        # 2. MTD bus stops
        if config.get('include_mtd', True):
            try:
                bus_stops = await bus_stops_within_1km(lat, lon)
                enriched_data['bus_stops_1km'] = bus_stops
                print(f"    ✓ Bus stops: {bus_stops}")
            except Exception as e:
                print(f"    ✗ MTD API error: {e}")
                enriched_data['bus_stops_1km'] = None
        
        # 3. Police beat
        if config.get('include_police_beat', True):
            try:
                beat = await police_beat_for_point(lat, lon)
                enriched_data['police_beat'] = beat
                print(f"    ✓ Police beat: {beat}" if beat else "    - No police beat")
            except Exception as e:
                print(f"    ✗ Police Beats API error: {e}")
                enriched_data['police_beat'] = None
        
        # 4. Google Places - Count nearby amenities
        if config.get('include_places', True):
            place_types = config.get('place_types', {
                'restaurants': ['restaurant'],
                'cafes': ['cafe'],
                'schools': ['school'],
                'parks': ['park'],
                'gyms': ['gym'],
                'supermarkets': ['supermarket', 'supermarket'],
            })
            
            for name, types in place_types.items():
                try:
                    count = await count_places_nearby(
                        lat, lon, 
                        included_types=types, 
                        radius_m=config.get('places_radius_m', 1000)
                    )
                    enriched_data[f'{name}_nearby'] = count
                    print(f"    ✓ {name.capitalize()}: {count}")
                except Exception as e:
                    print(f"    ✗ Places API error ({name}): {e}")
                    enriched_data[f'{name}_nearby'] = None
        
        # 5. Google Routes - Travel time to landmarks
        if config.get('include_routes', True):
            landmarks = config.get('landmarks', LANDMARKS)
            
            for landmark_name, landmark_coords in landmarks.items():
                try:
                    travel_time = await drive_time_minutes(
                        origin=(lat, lon),
                        dest=landmark_coords,
                        mode="DRIVE"
                    )
                    col_name = f'drive_to_{landmark_name.lower().replace(" ", "_")}_min'
                    enriched_data[col_name] = travel_time
                    if travel_time:
                        print(f"    ✓ Drive to {landmark_name}: {travel_time:.1f} min")
                    else:
                        print(f"    - No route to {landmark_name}")
                except Exception as e:
                    print(f"    ✗ Routes API error ({landmark_name}): {e}")
                    col_name = f'drive_to_{landmark_name.lower().replace(" ", "_")}_min'
                    enriched_data[col_name] = None
    
    except Exception as e:
        print(f"  ✗ Unexpected error enriching listing: {e}")
    
    return enriched_data


async def enrich_listings_batch(
    df: pd.DataFrame, 
    config: Dict[str, Any],
    batch_size: int = 10,
    start_idx: int = 0
) -> pd.DataFrame:
    """
    Enrich all listings in the dataframe with API data.
    
    Args:
        df: DataFrame with housing listings
        config: Configuration dict
        batch_size: Number of listings to process in parallel (be mindful of API rate limits)
        start_idx: Starting index (useful for resuming failed runs)
    
    Returns:
        DataFrame with enriched data
    """
    total = len(df)
    print(f"\n{'='*60}")
    print(f"Starting enrichment of {total} listings...")
    print(f"{'='*60}\n")
    
    # Create a copy to avoid modifying original
    enriched_df = df.copy()
    
    # Process in batches
    for i in range(start_idx, total, batch_size):
        batch_end = min(i + batch_size, total)
        print(f"\n--- Processing batch {i//batch_size + 1}: listings {i+1} to {batch_end} ---\n")
        
        # Create tasks for this batch
        tasks = []
        for idx in range(i, batch_end):
            row = df.iloc[idx]
            print(f"[{idx+1}/{total}] {row.get('address', 'Unknown address')}")
            tasks.append(enrich_listing(row, config))
        
        # Execute batch in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Add results to dataframe
        for idx_offset, result in enumerate(results):
            idx = i + idx_offset
            if isinstance(result, Exception):
                print(f"  ✗ Error processing listing {idx+1}: {result}")
                continue
            
            # Add each enriched field to the dataframe
            for col, value in result.items():
                enriched_df.at[idx, col] = value
        
        # Save intermediate results every batch (in case of failures)
        if config.get('save_intermediate', True):
            intermediate_path = config.get('intermediate_path', 'data/listings_intermediate.csv')
            enriched_df.to_csv(intermediate_path, index=False)
            print(f"\n  💾 Saved intermediate results to {intermediate_path}")
        
        # Small delay between batches to avoid rate limits
        if batch_end < total:
            await asyncio.sleep(config.get('batch_delay_seconds', 1))
    
    print(f"\n{'='*60}")
    print(f"✅ Enrichment complete!")
    print(f"{'='*60}\n")
    
    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description='Enrich housing listings CSV with API-derived attributes'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/listings_details_allcities.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/listings_enriched.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=5,
        help='Number of listings to process in parallel (default: 5)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index for resuming interrupted runs (default: 0)'
    )
    parser.add_argument(
        '--skip-census',
        action='store_true',
        help='Skip Census API calls'
    )
    parser.add_argument(
        '--skip-mtd',
        action='store_true',
        help='Skip MTD API calls'
    )
    parser.add_argument(
        '--skip-police-beat',
        action='store_true',
        help='Skip Police Beat API calls'
    )
    parser.add_argument(
        '--skip-places',
        action='store_true',
        help='Skip Google Places API calls'
    )
    parser.add_argument(
        '--skip-routes',
        action='store_true',
        help='Skip Google Routes API calls'
    )
    parser.add_argument(
        '--places-radius',
        type=int,
        default=1000,
        help='Radius in meters for Google Places searches (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load data
    print(f"\n📂 Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} listings")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Verify required columns
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Configure enrichment
    config = {
        'include_census': not args.skip_census,
        'include_mtd': not args.skip_mtd,
        'include_police_beat': not args.skip_police_beat,
        'include_places': not args.skip_places,
        'include_routes': not args.skip_routes,
        'places_radius_m': args.places_radius,
        'save_intermediate': True,
        'intermediate_path': args.output.replace('.csv', '_intermediate.csv'),
        'batch_delay_seconds': 1,
    }
    
    # Show configuration
    print("\n⚙️  Enrichment configuration:")
    print(f"  • Census API: {'✓' if config['include_census'] else '✗'}")
    print(f"  • MTD API: {'✓' if config['include_mtd'] else '✗'}")
    print(f"  • Police Beats API: {'✓' if config['include_police_beat'] else '✗'}")
    print(f"  • Google Places API: {'✓' if config['include_places'] else '✗'}")
    print(f"  • Google Routes API: {'✓' if config['include_routes'] else '✗'}")
    print(f"  • Places search radius: {config['places_radius_m']}m")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Starting from index: {args.start_idx}")
    
    # Run enrichment
    try:
        enriched_df = asyncio.run(
            enrich_listings_batch(
                df, 
                config, 
                batch_size=args.batch_size,
                start_idx=args.start_idx
            )
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Intermediate results may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    print(f"\n💾 Saving enriched data to {args.output}...")
    try:
        enriched_df.to_csv(args.output, index=False)
        print(f"✓ Saved {len(enriched_df)} enriched listings")
        
        # Show summary of new columns
        original_cols = set(df.columns)
        new_cols = [col for col in enriched_df.columns if col not in original_cols]
        if new_cols:
            print(f"\n📊 Added {len(new_cols)} new columns:")
            for col in new_cols:
                non_null = enriched_df[col].notna().sum()
                print(f"  • {col}: {non_null}/{len(enriched_df)} values ({non_null/len(enriched_df)*100:.1f}%)")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        sys.exit(1)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    # Default configuration for quick start
    # You can run this script directly without any arguments:
    #   python code/enrich_listings.py
    # 
    # Or customize with command line arguments:
    #   python code/enrich_listings.py -i data/your_file.csv -o data/output.csv --skip-places
    
    main()

