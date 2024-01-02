from pydantic_numpy.model import NumpyModel
from pydantic import Field
import pydantic_numpy.typing as pnd
import numpy as np


class TestEQ:
    def test_eq_multilevel(self) -> None:
        class A(NumpyModel):
            a: pnd.NpNDArray = Field(frozen=True)
        
        class B(NumpyModel):
            a: A = Field(frozen=True)
        
        class C(NumpyModel):
            b: B = Field(frozen=True)
        
        a = A(a=np.zeros((2)))
        b = B(a=a)
        c = C(b=b)

        other_c = C(b=B(a=A(a=np.zeros((1)))))
        assert c == other_c