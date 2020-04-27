import inspect
from logger import logger

class Test:
    def __init__(self, a: int, b: int, offset: int=None):
        self.a = a
        self.b = b
        self.c = a + b
        self.offset = offset
        self.c = self.c + self.offset if self.offset is not None else self.c
    
    def __str__(self) -> str:
        return f'Test(a={self.a}, b={self.b}, c={self.c})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_constructor_dict(self) -> dict:
        result = {}
        for key, val in self.__dict__.items():
            logger.cyan(f'{key} in {self.get_constructor_params()}: {key in self.get_constructor_params()}')
            if key in self.get_constructor_params():
                result[key] = val
        return result

    def to_dict(self) -> dict:
        result = {}
        for key, val in self.__dict__.items():
            logger.cyan(f'{key} in {self.get_constructor_params()}: {key in self.get_constructor_params()}')
            if key in self.get_constructor_params():
                result[key] = val
        return result

    @classmethod
    def get_constructor_params(cls) -> list:
        return [param for param in list(inspect.signature(cls.__init__).parameters.keys()) if param != 'self']

test = Test(4, 5)
print(test)
print(f'test.get_constructor_params(): {test.get_constructor_params()}')
print(f'type(test.get_constructor_params()): {type(test.get_constructor_params())}')
print(f'test.to_dict(): {test.to_dict()}')
logger.yellow(inspect.signature(Test.__init__).parameters.keys())