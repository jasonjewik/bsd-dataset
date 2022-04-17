from typing import Tuple
from dataclasses import dataclass

longitude, latitude = float, float
Coordinate = Tuple[longitude, latitude]

@dataclass
class Region:
    top_left: Coordinate
    bottom_right: Coordinate

SouthAmerica = Region(top_left=(-90, 20), bottom_right=(-20, -50))
CentralAmerica = Region(top_left=(-120, 30), bottom_right=(-30, -15))
NorthAmerica = Region(top_left=(-140, 50), bottom_right=(-60, 20))
Mediterranean = Region(top_left=(-20, 50), bottom_right=(45, 30))
Africa = Region(top_left=(-20, 40), bottom_right=(60, -40))
SouthAsia = Region(top_left=(30, 40), bottom_right=(110, -10))
EastAsia = Region(top_left=(70, 50), bottom_right=(165, 0))
CentralAsia = Region(top_left=(40, 50), bottom_right=(120, 20))
MENA = Region(top_left=(-20, 40), bottom_right=(70, 10))
Australasia = Region(top_left=(110, 0), bottom_right=(180, -50))
SoutheastAsia = Region(top_left=(80, 25), bottom_right=(150, -15))

regions = [
    SouthAmerica,
    CentralAmerica,
    NorthAmerica,
    Mediterranean,
    Africa,
    SouthAsia,
    EastAsia,
    CentralAsia,
    MENA,
    Australasia,
    SoutheastAsia
]