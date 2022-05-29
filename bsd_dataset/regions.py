from typing import Optional, Tuple

# TODO @jasonjewik: If the user defines a region that wraps around the
# boundary of the map (e.g. from 170 degrees east to -170 degrees east),
# they currently cannot plot the region with show_map nor can they
# extract the data.


class Coordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def __getitem__(self, idx):
        if idx == 0:
            return self.latitude
        elif idx == 1:
            return self.longitude
        else:
            raise IndexError()

    def __repr__(self):
        return f'({self.latitude}, {self.longitude})'

    def get_latlon(self):
        return (self.latitude, self.longitude)

    def get_lonlat(self):
        return (self.longitude, self.latitude)

class Region:
    last_used_name = 0

    def __init__(self, top_left: Coordinate, bottom_right: Coordinate, longitude_range: int, name: Optional[str] = None):        
        self.top_left = top_left
        self.bottom_right = bottom_right
        if longitude_range != 180 and longitude_range != 360:
            raise ValueError(f'longitude range must be 180 or 360')
        self.longitude_range = longitude_range

        if name is None:
            Region.last_used_name += 1
            self.name = f'Region {Region.last_used_name}'            
        else:
            self.name = name

    def __str__(self):
        result = (
            f'Name: {self.name}\n'
            f'Top left: {self.top_left}\n'
            f'Bottom right: {self.bottom_right}\n'
            f'Longitude range: {self.longitude_range}'
        )
        return result

    def lon180_to_lon360(self, lon: float) -> float:
        return (lon + 360) % 360

    def lon360_to_lon180(self, lon: float) -> float:
        return ((lon + 180) % 360) - 180

    def as180(self):
        """Does not modify in-place."""
        if self.longitude_range == 180:
            return self
        tl = Coordinate(
            self.top_left.latitude,
            self.lon360_to_lon180(self.top_left.longitude))
        br = Coordinate(
            self.bottom_right.latitude,
            self.lon360_to_lon180(self.bottom_right.longitude))
        new_region = Region(
            name=self.name,
            top_left=tl,
            bottom_right=br,
            longitude_range=180)
        return new_region

    def as360(self):
        """Does not modify in-place."""
        if self.longitude_range == 360:
            return self
        tl = Coordinate(
            self.top_left.latitude,
            self.lon180_to_lon360(self.top_left.longitude))
        br = Coordinate(
            self.bottom_right.latitude,
            self.lon180_to_lon360(self.bottom_right.longitude))
        new_region = Region(
            name=self.name,
            top_left=tl,
            bottom_right=br,
            longitude_range=180)
        return new_region

    def get_latitudes(self) -> Tuple[float, float]:
        return tuple([self.top_left.latitude, self.bottom_right.latitude])

    def get_longitudes(self, out_longitude_range: Optional[int] = None) -> Tuple[float, float]:
        lons = [self.top_left.longitude, self.bottom_right.longitude]
        if out_longitude_range is None:
            # Return the longitude coordinates in their original range.
            return tuple(lons)
        elif out_longitude_range != 180 and out_longitude_range != 360:
            raise ValueError(f'out longitude range must be 180 or 360')
        elif self.longitude_range == 360 and out_longitude_range == 180:
            # Stored longitudes are in [0, 360] but the user wants them in
            # [-180, 180].
            lons = [self.lon360_to_lon180(lon) for lon in lons]
        elif out_longitude_range == 360 and self.longitude_range == 180:
            # Stored longitudes are in [-180, 180] but the user wants them in
            # [0, 360].
            lons = [self.lon180_to_lon360(lon) for lon in lons]
        return tuple(lons)


# Pre-defined regions that roughly match CORDEX domains.
# https://cordex.org/wp-content/uploads/2012/11/CORDEX-domain-description_231015.pdf
# Most CORDEX domains do not have clean boundaries that lie along latitude or
# longitude lines, but here we define rectangular-shaped regions purely using
# top left and bottom right coordinates for simplicity.

SouthAmerica = Region(
        name='South America',
        top_left=Coordinate(20, 270),
        bottom_right=Coordinate(-55, 330),
        longitude_range=360)

CentralAmerica = Region(
    name='Central America',
    top_left=Coordinate(30, 240),
    bottom_right=Coordinate(-15, 335),
    longitude_range=360)
    
NorthAmerica = Region(
    name='North America',
    top_left=Coordinate(65, 220),
    bottom_right=Coordinate(15, 300),
    longitude_range=360)
    
Europe = Region(
    name='Europe',
    top_left=Coordinate(65, 350),
    bottom_right=Coordinate(30, 40),
    longitude_range=360)

Africa = Region(
    name='Africa',
    top_left=Coordinate(40, 335),
    bottom_right=Coordinate(-40, 60),
    longitude_range=360)

SouthAsia = Region(
    name='South Asia',
    top_left=Coordinate(45, 25),
    bottom_right=Coordinate(-15, 110),
    longitude_range=360)

EastAsia = Region(
    name='East Asia',
    top_left=Coordinate(55, 70),
    bottom_right=Coordinate(5, 150),
    longitude_range=360)

CentralAsia = Region(
    name='Central Asia',
    top_left=Coordinate(60, 40),
    bottom_right=Coordinate(20, 120),
    longitude_range=360)

Australasia = Region(
    name='Australasia',
    top_left=Coordinate(8, 100),
    bottom_right=Coordinate(-49, 179),
    longitude_range=360)

Mediterranean = Region(
    name='Mediterranean',
    top_left=Coordinate(55, 345),
    bottom_right=Coordinate(25, 45),
    longitude_range=360)

MENA = Region(
    name='Middle East and North Africa',
    top_left=Coordinate(45, 340),
    bottom_right=Coordinate(-7, 76),
    longitude_range=360)

SoutheastAsia = Region(
    name='Southeast Asia',
    top_left=Coordinate(30, 90),
    bottom_right=Coordinate(-15, 147),
    longitude_range=360)

# Region for testing
California = Region(
    name='California',
    top_left=Coordinate(42, -124),
    bottom_right=Coordinate(32, -114),
    longitude_range=180)

regions = [
    SouthAmerica,
    CentralAmerica,
    NorthAmerica,
    Europe,
    Africa,
    SouthAsia,
    EastAsia,
    CentralAsia,
    Australasia,
    Mediterranean,
    MENA,
    SoutheastAsia,
    California
]