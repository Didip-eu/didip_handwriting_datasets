import pytest
import sys
from pathlib import Path
from torch import Tensor

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from handwriting_datasets import monasterium

@pytest.fixture(scope="module")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


def test_load_from_tsv( data_path ):
    ds = monasterium.MonasteriumDataset(task='htr',shape='bbox', from_tsv_file=data_path.joinpath('20_ms.tsv'))
    assert len(ds)==20
    assert type(ds[0][0]) is Tensor
    assert type(ds[0][1]) is str

def test_dummy( data_path ):
    assert monasterium.dummy()
    assert isinstance(data_path, Path )



