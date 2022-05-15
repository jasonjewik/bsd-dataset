from bsd_dataset.datasets.download_utils import select_periods

class TestGetPeriods:
    def test_simple(self):
        periods = [
            ('2001-01-01', '2001-12-31'),
            ('2002-01-01', '2002-12-31'),
            ('2003-01-01', '2003-12-31'),
            ('2004-01-01', '2004-12-31'),
            ('2005-01-01', '2005-12-31')
        ]
        selected, success = select_periods(
            start='2001-01-01',
            end='2004-12-31',
            periods=periods
        )
        expected = [
            ('2001-01-01', '2001-12-31'),
            ('2002-01-01', '2002-12-31'),
            ('2003-01-01', '2003-12-31'),
            ('2004-01-01', '2004-12-31')
        ]
        assert success == True
        assert selected == expected        

    def test_no_match(self):
        periods = [
            ('2001-01-01', '2001-12-31'),
            ('2002-01-01', '2002-12-31'),
            ('2003-01-01', '2003-12-31'),
            ('2004-01-01', '2004-12-31'),
            ('2005-01-01', '2005-12-31')
        ]
        _, success = select_periods(
            start='1980-01-01',
            end='1990-12-31',
            periods=periods
        )
        assert success == False

    def test_empty_periods(self):
        periods = []
        _, success = select_periods(
            start='1980-01-01',
            end='1990-12-31',
            periods=periods
        )
        assert success == False

    def test_nonaligned_ends(self):
        periods = [
            ('2001-01-01', '2001-12-31'),
            ('2002-01-01', '2002-12-31'),
            ('2003-01-01', '2003-12-31'),
            ('2004-01-01', '2004-12-31'),
            ('2005-01-01', '2005-12-31')
        ]
        selected, success = select_periods(
            start='2001-06-10',
            end='2004-01-20',
            periods=periods
        )
        expected = [
            ('2001-01-01', '2001-12-31'),
            ('2002-01-01', '2002-12-31'),
            ('2003-01-01', '2003-12-31'),
            ('2004-01-01', '2004-12-31'),
        ]
        assert success == True
        assert selected == expected        

    def test_period_gap(self):
        periods = [
            ('2001-01-01', '2001-12-31'),
            ('2004-01-01', '2004-12-31'),
            ('2005-01-01', '2005-12-31')
        ]
        selected, success = select_periods(
            start='2001-06-10',
            end='2004-01-20',
            periods=periods
        )
        expected = [
            ('2001-01-01', '2001-12-31'),
            ('2004-01-01', '2004-12-31'),
        ]
        assert success == True
        assert selected == expected
