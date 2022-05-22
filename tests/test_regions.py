from bsd_dataset.regions import Coordinate, Region, regions

class TestRegion:
    def test_names(self):
        # If a name is not specified, the default naming convention should be
        # used, which is just "Region n", where n is the least unused natural
        # number.
        r1 = Region(
            top_left=Coordinate(100, 100),
            bottom_right=Coordinate(100, 100),
            longitude_range=360)
        assert r1.name == 'Region 1'
        r2 = Region(
            top_left=Coordinate(100, 100),
            bottom_right=Coordinate(100, 100),
            longitude_range=360)
        assert r2.name == 'Region 2'

    def test_get_latslons(self):
        # Test that we get out what we put in.
        r1 = Region(
            top_left=Coordinate(20, 270),
            bottom_right=Coordinate(-55, 330),
            longitude_range=360)
        assert r1.get_latitudes() == (20, -55)
        assert r1.get_longitudes() == (270, 330)

    def test_invalid_longitude_range(self):
        # Longitude range has to be either 180 or 360.
        try:
            Region(
                top_left=Coordinate(100, 100),
                bottom_right=Coordinate(100, 100),
                longitude_range=270)
        except ValueError as e:
            return
        except Exception as e:
            raise e

    def test_lats_after_conversion(self):
        # After converting longitude range, the latitudes should remain the
        # same.
        for r in regions:
            lats = r.get_latitudes()
            r = r.as180()
            assert lats == r.get_latitudes()

    def test_get_lons(self):
        # Getting longitudes without specifying out range should return
        # the longitudes in the original range.
        for r in regions:
            if r.longitude_range == 360:
                assert r.get_longitudes() == r.get_longitudes(360), f'{r.name} failed'
            elif r.longitude_range == 180:
                assert r.get_longitudes() == r.get_longitudes(180), f'{r.name} failed'

    def test_lons_after_conversion1(self):
        # After converting to the same longitude range, the longitudes should
        # remain the same.
        for r in regions:
            lons = r.get_longitudes()
            if r.longitude_range == 360:            
                r = r.as360()                
            elif r.longitude_range == 180:
                r = r.as180()
            assert lons == r.get_longitudes(), f'{r.name} failed'

    def test_lons_after_conversion2(self):
        # After converting to a different longitude range, the longitudes
        # should be different.
        for r in regions:
            lons180 = r.get_longitudes(180)
            lons360 = r.get_longitudes(360)
            r = r.as180()
            assert lons180 == r.get_longitudes(), f'{r.name} failed'
            r = r.as360()
            assert lons360 == r.get_longitudes(360), f'{r.name} failed'