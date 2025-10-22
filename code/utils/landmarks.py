from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Landmark:
    name: str
    lat: float
    lon: float

# UIUC Main Quad (official page lists 40°06'36.88"N, 88°13'38.13"W)
UIUC_MAIN_QUAD = Landmark("UIUC Main Quad", 40.110244, -88.227258)

# Downtown Champaign centroid: near 201 N Neil St / 1 E Main St
DOWNTOWN_CHAMPAIGN = Landmark("Downtown Champaign", 40.117489, -88.243800)

# Dictionary format for easy iteration (name -> (lat, lon))
LANDMARKS: dict[str, Tuple[float, float]] = {
    "UIUC Main Quad": (40.110244, -88.227258),
    "Downtown Champaign": (40.117489, -88.243800),
    "Carle Hospital": (40.113056, -88.214444),
    "Memorial Stadium": (40.099167, -88.235833),
    "Willard Airport": (40.039167, -88.278056),
}
