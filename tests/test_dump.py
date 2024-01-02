import tempfile
from pydantic_numpy.model import NumpyModel
from pydantic import Field
import pydantic_numpy.typing as pnd
import numpy as np
from pathlib import Path


class TestDump:
    def test_multilevel_dump(self) -> None:
        class A(NumpyModel):
            a: pnd.NpNDArray = Field(frozen=True)
        
        class B(NumpyModel):
            a: A = Field(frozen=True)
        
        class C(NumpyModel):
            b: B = Field(frozen=True)
        
        a = A(a=np.zeros((2)))
        b = B(a=a)
        c = C(b=b)
        
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c.dump(path, "multilevel_model_test")
            
            new_c = C.load(path, "multilevel_model_test")
            assert new_c == c