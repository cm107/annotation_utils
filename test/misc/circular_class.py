from __future__ import annotations
from typing import List
import inspect

class Test:
    def __init__(self, a: int, b: int, test_list: List[Test]=None):
        self.a = a
        self.b = b
        self.test_list = test_list if test_list is not None else []

    @classmethod
    def get_constructor_params(cls) -> list:
        return [param for param in list(inspect.signature(cls.__init__).parameters.keys()) if param != 'self']

    def to_constructor_dict(self) -> dict:
        constructor_dict = {}
        for key, val in self.__dict__.items():
            if key in self.get_constructor_params():
                constructor_dict[key] = val
        return constructor_dict

    def __str__(self) -> str:
        constructor_dict = self.to_constructor_dict()
        param_str = ''
        for key, val in constructor_dict.items():
            if param_str == '':
                param_str += f'{key}={val}'
            else:
                param_str += f', {key}={val}'
        return f'Test({param_str})'
    
    def __repr__(self) -> str:
        return self.__str__()

test = Test(
    a=1, b=2,
    test_list=[
        Test(a=10, b=11),
        Test(a=20, b=21),
        Test(
            a=30, b=31,
            test_list=[
                Test(a=30, b=31),
                Test(a=40, b=41),
            ]
        )
    ]
)

print(test)