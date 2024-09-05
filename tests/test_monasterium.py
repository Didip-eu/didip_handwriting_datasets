import pytest
import sys
from pathlib import Path
import torch
from torch import Tensor
from torchvision.transforms import PILToTensor, ToPILImage
from functools import partial

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from handwriting_datasets import monasterium

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def data_set( data_path ):
    return monasterium.MonasteriumDataset(
            task='htr', shape='bbox',
            from_tsv_file=data_path.joinpath('20_ms.tsv'),
            transform=partial( monasterium.MonasteriumDataset.size_fit_transform, max_h=300, max_w=2000))


def test_size_fit_transform():
    img_to_resize = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.MonasteriumDataset.size_fit_transform( img_to_resize, 30, 200 )
    assert sample_dict['img'].shape == torch.Size([3,30,200])
    assert sample_dict['height'] == 30
    assert sample_dict['width'] == 150

def test_default_transform( data_set ):
    """Testing default transform"""
    img_to_resize = torch.randint(10,255, (3, 100, 500), dtype=torch.uint8)
    final_img = torch.zeros((3,300,2000))
    final_img[:,:100,:500]=img_to_resize

    timg = data_set.transform( ToPILImage()( img_to_resize ))
    #print("timg=", timg, " with type=", type(timg))
    assert len(timg) == 3
    assert timg['img'].equal( final_img )
    assert timg['height'] == 100
    assert timg['width'] == 500
    #assert timg[4] == 500


def test_load_from_tsv( data_path ):
    samples = monasterium.MonasteriumDataset.load_from_tsv( data_path.joinpath('20_ms.tsv'))
    assert len(samples) == 20


def test_dataset_from_tsv_item_type( data_set ):
    assert len(data_set)==20
    assert type(data_set[0]) is dict
    assert len(data_set[0]) == 4


def test_load_from_tsv_item_subtypes( data_set ):
    assert type(data_set[0]['img']) is Tensor # img tensor
    assert type(data_set[0]['transcription']) is str    # transcription
    assert type(data_set[0]['height']) is int    # img height (after resizing)
    assert type(data_set[0]['width']) is int    # img width  (after resizing)
    #assert type(data_set[0]['mask']) is Tensor    # img width  (after resizing)

def test_load_from_tsv_img_properties( data_set ):
    assert len(data_set)==20
    assert data_set[0]['img'].shape[1] == 300    # img height (after resizing)
    assert data_set[0]['img'].shape[2] == 2000    # img width  (after resizing)

def test_dummy( data_path ):
    assert monasterium.dummy()
    assert isinstance(data_path, Path )



