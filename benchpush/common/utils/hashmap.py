""" Original code from https://entitycrisis.blogspot.com/2007/11/spatial-hashing.html """


class HashMap(object):
    """
    Hashmap is a spatial variant of the standard hashmap.
    This class is essential in hashing finite precision
    spatial coordinates since using floats as keys in a
    standard hashmap is prone to round off error creating
    multiple keys that are very close in precision.
    """

    def __init__(self, cell_size, scale):
        self.cell_size = cell_size  # determines how close keys are for them to be considered the same
        self.scale = scale  # scales keys so more sig figs are considered
        self.scale_cell_ratio = self.scale / self.cell_size
        self._grid = {}

    def __len__(self):
        return len(self._grid)

    def __contains__(self, item):
        return self._key(item) in self._grid

    def to_dict(self):
        return dict(self._grid)

    def _key(self, point):
        return (
            round(point[0] * self.scale_cell_ratio) * self.cell_size,  # x
            round(point[1] * self.scale_cell_ratio) * self.cell_size,  # y
            round(point[2])                                            # heading
        )

    def add(self, point):
        """
        Insert point into the hashmap.
        """
        self._grid.setdefault(self._key(point), []).append(point)

    def query(self, point):
        """
        Return all objects in the cell specified by point.
        """
        assert point in self
        return self._grid.setdefault(self._key(point), [])

    def pop(self, point):
        """
        Remove point from hashmap.
        """
        return self._grid.pop(self._key(point))

    @classmethod
    def from_points(cls, cell_size, scale, points):
        """
        Build a HashMap from a list of points.
        """
        hashmap = cls(cell_size, scale)
        setdefault = hashmap._grid.setdefault
        key = hashmap._key
        for point in points:
            setdefault(key(point), []).append(point)
        return hashmap
