import pytest
import sys
from pathlib import Path
import torch
from torch import Tensor
from torchvision.transforms import PILToTensor, ToPILImage, Compose
from functools import partial

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from handwriting_datasets import monasterium

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


@pytest.fixture(scope="session")
def bbox_data_set( data_path ):
    return monasterium.MonasteriumDataset(
            task='htr', shape='bbox',
            from_tsv_file=data_path.joinpath('bbox', 'monasterium_ds_train.tsv'),
            transform=Compose([ monasterium.ResizeToMax(300,2000), monasterium.PadToSize(300,2000) ]))

@pytest.fixture(scope="session")
def polygon_data_set( data_path ):
    return monasterium.MonasteriumDataset(
            task='htr', shape='polygon',
            from_tsv_file=data_path.joinpath('polygons', 'monasterium_ds_train.tsv'),
            transform=Compose([ monasterium.ResizeToMax(300,2000), monasterium.PadToSize(300,2000) ]))

@pytest.mark.parametrize(
        "subset, set_length",
        [('train', 5), ('validate', 3), ('test', 2)])
def test_split_set( subset, set_length):
    """
    Given a list of ratios for 3 subsets, and a subset name, return 
    the proper subset, with the proper size
    """
    samples = [ {'a':1.2, 'b':(4,5), 'c':'some' } for i in range(10) ]
    s = monasterium.MonasteriumDataset.split_set( samples, (.5,.3,.2), subset )
    assert len(s) == set_length
    assert s[0]['b'] == (4,5)


def test_ResizeToMax_img_field():
    """
    Raw transform
    """
    img_to_resize = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.ResizeToMax(30, 200)( {'img': img_to_resize, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['img'].shape == torch.Size([3,30,150])

def test_ResizeToMax_heigh_width_fields():
    """
    Raw transform
    """
    img_to_resize = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.ResizeToMax(30, 200)( {'img': img_to_resize, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['height'] == 30
    assert sample_dict['width'] == 150


def test_ResizeToMax_transcription_field():
    """
    Raw transform
    """
    img_to_resize = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.ResizeToMax(30, 200)( {'img': img_to_resize, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['transcription'] == 'abc'

def test_PadToSize_img_field():
    """
    Raw transform
    """
    img_to_pad = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.PadToSize(200, 600)( {'img': img_to_pad, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['img'].shape == torch.Size([3,200,600])

def test_PadToSize_height_width_fields():
    """
    Raw transform
    """
    img_to_pad = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.PadToSize(200, 600)( {'img': img_to_pad, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['height'] == 100
    assert sample_dict['width'] == 500

def test_PadToSize_transcription_field():
    """
    Raw transform
    """
    img_to_pad = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.PadToSize(200, 600)( {'img': img_to_pad, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['transcription'] == 'abc'

def test_PadToSize_mask_field():
    """
    Raw transform
    """
    img_to_pad = torch.randint(10,255, (3, 100, 500))
    sample_dict = monasterium.PadToSize(200, 600)( {'img': img_to_pad, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    mask = torch.zeros((3,200,600), dtype=torch.bool)
    mask[:,:100,:500]=1
    assert sample_dict['mask'].shape == torch.Size( [3,200,600])
    assert sample_dict['mask'].dtype is torch.bool
    assert sample_dict['mask'].equal( mask )

def test_PadToSize_mask_field_handling_holes():
    """
    Raw transform
    """
    img_to_pad = torch.randint(10,255, (3, 100, 500))
    img_to_pad[:,30:60, 2:10]=0
    sample_dict = monasterium.PadToSize(200, 600)( {'img': img_to_pad, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    mask = torch.zeros((3,200,600), dtype=torch.bool)
    mask[:,:100,:500]=1
    assert sample_dict['mask'].shape == torch.Size( [3,200,600])
    assert sample_dict['mask'].dtype is torch.bool
    assert sample_dict['mask'].equal( mask )


def test_ResizePadCompose():
    """ Resize and Pad """
    img_to_resize = torch.randint(10,255, (3, 100, 500))
    sample_dict = Compose([ 
            monasterium.ResizeToMax(30, 200),
            monasterium.PadToSize(200, 600)])( {'img': img_to_resize, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    assert sample_dict['img'].shape == torch.Size([3,200,600])
    assert sample_dict['height'] == 30
    assert sample_dict['width'] == 150
    assert sample_dict['transcription'] == 'abc'




def test_default_transform( bbox_data_set ):
    """Default transform (torch wrapper)
    """
    img_to_resize = torch.randint(10,255, (3, 100, 500), dtype=torch.uint8)
    final_img = torch.zeros((3,300,2000))
    final_img[:,:100,:500]=img_to_resize

    sample = bbox_data_set.transform( {'img': img_to_resize, 'height': 100, 'width': 500, 'transcription': 'abc'} )
    #print("sample=", timg, " with type=", type(timg))
    assert len(sample) == 5
    assert sample['img'].equal( final_img )
    assert sample['height'] == 100
    assert sample['width'] == 500
    assert sample['transcription'] == 'abc'
    assert type(sample['mask']) is Tensor
    
def test_getitem_bbox( bbox_data_set ):
    sample = bbox_data_set[0]
    assert len(sample) == 5
    assert type(sample['img']) is Tensor
    assert type(sample['transcription']) is str
    assert type(sample['height']) is int
    assert type(sample['width']) is int
    assert type(sample['mask']) is Tensor

def test_getitem_polygons( polygon_data_set ):
    sample = polygon_data_set[0]
    print(sample)
    assert len(sample) == 5
    assert type(sample['img']) is Tensor
    assert type(sample['transcription']) is str
    assert type(sample['height']) is int
    assert type(sample['width']) is int
    assert type(sample['mask']) is Tensor
    assert type(sample['polygon_mask']) is Tensor

def test_load_from_tsv_bbox( data_path ):
    samples = monasterium.MonasteriumDataset.load_from_tsv( data_path.joinpath('bbox', 'monasterium_ds_train.tsv'))
    assert len(samples) == 20

def test_load_from_tsv_polygons( data_path ):
    samples = monasterium.MonasteriumDataset.load_from_tsv( data_path.joinpath('polygons', 'monasterium_ds_train.tsv'))
    assert len(samples) == 20

def test_dataset_from_tsv_item_type_bbox( bbox_data_set ):
    assert len(bbox_data_set)==20
    assert type(bbox_data_set[0]) is dict
    assert len(bbox_data_set[0]) == 6


def test_dataset_from_tsv_item_type_polygons( polygon_data_set ):
    assert len(polygon_data_set)==20
    assert type(polygon_data_set[0]) is dict
    assert len(polygon_data_set[0]) == 6

def test_load_from_tsv_item_subtypes_bbox( bbox_data_set ):
    assert type(bbox_data_set[0]['img']) is Tensor # img tensor
    assert type(bbox_data_set[0]['transcription']) is str    # transcription
    assert type(bbox_data_set[0]['height']) is int    # img height (after resizing)
    assert type(bbox_data_set[0]['width']) is int    # img width  (after resizing)
    assert type(bbox_data_set[0]['mask']) is Tensor    # img width  (after resizing)

def test_load_from_tsv_item_subtypes_polygons( polygon_data_set ):
    assert type(polygon_data_set[0]['img']) is Tensor # img tensor
    assert type(polygon_data_set[0]['transcription']) is str    # transcription
    assert type(polygon_data_set[0]['height']) is int    # img height (after resizing)
    assert type(polygon_data_set[0]['width']) is int    # img width  (after resizing)
    assert type(polygon_data_set[0]['mask']) is Tensor    # img width  (after resizing)

def test_load_from_tsv_img_properties_bbox( bbox_data_set ):
    assert len(bbox_data_set)==20
    assert bbox_data_set[0]['img'].shape[1] == 300    # img height (after resizing)
    assert bbox_data_set[0]['img'].shape[2] == 2000    # img width  (after resizing)

def test_load_from_tsv_img_properties_polygons( polygon_data_set ):
    assert len(polygon_data_set)==20
    assert polygon_data_set[0]['img'].shape[1] == 300    # img height (after resizing)
    assert polygon_data_set[0]['img'].shape[2] == 2000    # img width  (after resizing)

def test_dummy( data_path ):
    assert monasterium.dummy()
    assert isinstance(data_path, Path )



