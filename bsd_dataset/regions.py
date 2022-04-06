from typing import Tuple
from dataclasses import dataclass

longitude, latitude = float, float
Coordinate = Tuple[longitude, latitude]

@dataclass
class RegionCoordinates:
    top_left_corner: Coordinate
    center_north_boundary: Coordinate
    top_right_corner: Coordinate
    center_east_boundary: Coordinate
    center_point: Coordinate
    center_west_boundary: Coordinate
    bottom_left_corner: Coordinate
    center_south_boundary: Coordinate
    bottom_right_corner: Coordinate

# TODO: Currently, only top_left_corner and bottom_right_corner are used.
# Therefore, the other coordinates are set to (-1, -1). Eventually, we might
# want to use these, to conform to CORDEX standard.
NorthAmerica = RegionCoordinates(
    top_left_corner=(-125, 30),
    center_north_boundary=(-1, -1),
    top_right_corner=(-1, -1),
    center_east_boundary=(-1, -1),
    center_point=(-1, -1),
    center_west_boundary=(-1, -1),
    bottom_left_corner=(-1, -1),
    center_south_boundary=(-1, -1),
    bottom_right_corner=(-75, 50)
)

regions = [
    NorthAmerica
]