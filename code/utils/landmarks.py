from dataclasses import dataclass

@dataclass(frozen=True)
class Landmark:
    name: str
    lat: float
    lon: float

# UIUC Main Quad (official page lists 40°06'36.88"N, 88°13'38.13"W)
UIUC_MAIN_QUAD = Landmark("UIUC Main Quad", 40.110244, -88.227258)

# Downtown Champaign centroid: near 201 N Neil St / 1 E Main St
DOWNTOWN_CHAMPAIGN = Landmark("Downtown Champaign", 40.117489, -88.243800)
