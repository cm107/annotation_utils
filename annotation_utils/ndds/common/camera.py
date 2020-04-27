from typing import List

class CameraParam:
    def __init__(self, f: List[float], c: List[float], T: List[float], resx: float, resy: float):
        self.resx = resx
        self.resy = resy
        self.f = f
        self.c = c
        self.T = T
    
    def __str__(self) -> str:
        return f"camera intrinsics ({self.f},{self.c}, {self.T}), resolution x,y: {self.resx}, {self.resy}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return self.__dict__

    def to_dict_fct(self) -> dict:
        return {"f": self.f, "c": self.c, "T": self.T}