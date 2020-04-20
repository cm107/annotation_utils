class BBox:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.xmin = x
        self.ymin = y
        self.xmax = x + width
        self.ymax = y + height
        self.area = width * height

    def __str__(self):
        return f"BBox: (x,y)=({self.x},{self.y}), (width,height)=({self.width},{self.height})"

    def __repr__(self):
        return self.__str__()

    def get_minmax(self) -> list:
        return [self.xmin, self.ymin, self.xmax, self.ymax]