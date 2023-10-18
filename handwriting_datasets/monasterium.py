import sys
from typing import *
import defusedxml.ElementTree as ET
from pathlib import *
import warnings
from PIL import Image, ImageDraw
from PIL import ImagePath as IP
import re
import argparse
import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import logging
import random
import shutil as sh

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2




"""
Generate image/GT pairs for Monasterium transcriptions. This is a multi-use, generic dataset class, that can be used to generate

- line-segmentation dataset
- HTR dataset: Extract line polygons from each image and their respective transcriptions from the PageXML file

"""

logging.basicConfig( level=logging.DEBUG )

logger = logging.getLogger()

class MonasteriumDataset(VisionDataset):

    def __init__(
                self,
                basefolder: str,
                subset: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract: bool = True,
                segmentation_only = False,
                target_folder: str ='line_imgs',
                limit: int = 0,
                ):
        """
        Args:
            basefolder (str): Where the original files (pageXML documents and page images) are stored
            subset (str): 'train' (default), 'validate' or 'test'.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground truth at loading time.
            extract: if True (default), extract and store line images from the pages (default); 
                     otherwise try loading the data from an existing CSV file and return.
            segmentation_only: if True, only generates PageXML files that share their prefix with the images (note that those MD5 aren not the same as FSDB)
            target_folder: Where line images and ground truth transcriptions are created (assumed to 
                           be relative to the caller's pwd).
            limit (int): Stops after extracting {limit} images (for testing purpose only).
        """

        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        csv_path = Path(target_folder, 'monasterium_ds.csv')

        super().__init__(basefolder, transform=trf, target_transform=target_transform )
        self.setname = 'Monasterium'
        self.basefolder=basefolder

        # input PageXML files
        self.pagexmls = Path(self.basefolder).glob('*.xml')

        self.data = []

        if segmentation_only:
            self.extract_pages( target_folder )
            return

        # when DS not created from scratch, try loading from an existing CSV file
        if not extract:
            if csv_path.exists():
                self.data = self.load_from_csv( csv_path )
            return

        self.data = self.split_set( self.extract_lines( target_folder, limit=limit ), subset )

        # Generate a CSV file with one entry per img/transcription pair
        self.dump_to_csv( csv_path, self.data )


    def purge(self, folder: str) -> int:
        """
        Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the CSV file.

        Args:
            folder (str): Name of the subfolder to purge (relative the caller's
                          pwd
        """
        cnt = 0
        for item in Path( folder ).iterdir():
            item.unlink()
            cnt += 1
        return cnt

    def dump_to_csv(self, file_path: Path, data: list):
        """
        Create a CSV file with all pairs (<line image path>, transcription).

        Args:
            file_path (pathlib.Path): A file path (relative to the caller's pwd).
        """
        with open( file_path, 'w' ) as of:
            for path, gt in self.data:
                #print('{}\t{}'.format( path, gt ))
                of.write( '{}\t{}'.format( path, gt ) )


    def load_from_csv(self, file_path: Path) -> list:
        """
        Load pairs (<line image path>, transcription) from an existing CSV file.

        Args:
            file_path (pathlib.Path): A file path (relative to the caller's pwd).

        Returns:
            list: A list of 2-tuples whose first element is the (relative) path of
                  the line image and the second element is the transcription.
        """
        with open( file_path, 'r') as infile:
            return [ pair[:-1].split('\t') for pair in infile ]



    def extract_lines(self, target_folder: str, shape='polygons', text_only=False, limit=0) -> List[Tuple[str, str]]:
        """
        Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        Args:
            target_folder (str): Line images are extracted in this subfolder (relative to the caller's pwd).
            shape (str): Extract lines as polygon-within-bbox (default) or as bboxes.
            text_only (bool): Store only the transcriptions (*.gt files).
            limit (int): Stops after extracting {limit} images (for testing purpose).

        Returns:
            list: An array of pairs (img_file_path, transcription)

        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( target_folder ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( target_folder ) # ensure there are no pre-existing line items in the target directory

        gt_lengths = []
        img_sizes = []
        count = 0 # for testing purpose

        items = []

        for page in self.pagexmls:

            xml_id = Path( page ).stem
            img_path = Path(self.basefolder, f'{xml_id}.jpg')
            print( img_path )

            with open(page, 'r') as page_file:

                page_tree = ET.parse( page_file )

                ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

                page_root = page_tree.getroot()

                if not text_only:

                    try:
                        page_image = Image.open( img_path, 'r')
                    except Image.DecompressionBombWarning as dcb:
                        logger.debug( f'{dcb}: ignoring page' )
                        continue

                for textline_elt in page_root.findall( './/pc:TextLine', ns ):
                    if limit and count == limit:
                        return items

                    textline = dict()
                    textline['id']=textline_elt.get("id")
                    textline['transcription']=textline_elt.find('./pc:TextEquiv', ns).find('./pc:Unicode', ns).text

                    # skip lines that don't have a transcription
                    if not textline['transcription']:
                        continue

                    polygon_string=textline_elt.find('./pc:Coords', ns).get('points')
                    coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                    textline['bbox'] = IP.Path( coordinates ).getbbox()
                    img_sizes.append( [ textline['bbox'][i+2]-textline['bbox'][i] for i in (0,1) ])
                    img_path_prefix = Path(target_folder, f"{xml_id}-{textline['id']}" )
                    textline['img_path'] = img_path_prefix.with_suffix('.png')

                    items.append( (textline['img_path'], textline['transcription']) ) 
                    if not text_only:
                        bbox_img = page_image.crop( textline['bbox'] )

                        if shape=='bbox':
                            bbox_img.save( textline['img_path'] )

                        else:
                            # mask (=line polygon)
                            mask = Image.new("L", bbox_img.size, 0)
                            drawer = ImageDraw.Draw(mask)
                            leftx, topy = textline['bbox'][:2]
                            transposed_coordinates = [ (x-leftx, y-topy) for x,y in coordinates ]
                            drawer.polygon( transposed_coordinates, fill=255 )

                            # background
                            bg = Image.new("RGB", bbox_img.size, color=0)

                            # composite
                            Image.composite(bbox_img, bg, mask).save( Path( textline['img_path'] ))

                    with open( img_path_prefix.with_suffix('.gt'), 'w') as gt_file:
                        gt_file.write( textline['transcription'] )
                        gt_lengths.append(len( textline['transcription']))

                    count += 1

        return items


    def split_set(self, pairs: Tuple[str, str], subset: str = 'train') -> List[ Tuple[int, int]]:
        """
        Split a set of pairs (<line image>, transcription>) into 3 sets: train (70%),
        validation (10%), and test (20%).

        Args:
            pairs (tuple): A list of all pairs (<line image>, transcription>) in the dataset.
            subset (str): Subset to return: 'train' (default), 'validate', or 'test'.

        Returns:
            list: A list of pairs, either for training, validation, or testing.
        """
        random.seed(10)
        test_count = int( len(pairs)*.2 )
        validation_count = int( len(pairs)*.1 )

        all_pairs = set( pairs )
        test_pairs = set( random.sample( all_pairs, test_count) )
        all_pairs -= test_pairs

        validation_pairs = set( random.sample( all_pairs, validation_count))
        all_pairs -= validation_pairs

        if subset == 'validate':
            return validation_pairs
        if subset == 'test':
            return test_pairs
        return list( all_pairs )


    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            tuple: A pair (image, transcription)
        """
        transcr = self.data[index][1]
        img = self.transform( Image.open(self.data[index][0], 'r') )
        return (img, transcr)


    def __len__(self) -> int:
        """
        Returns:
            tuple: A pair (image, transcription)
        """
        return len( self.data )



    def count_line_items(self, folder) -> Tuple[int, int]:
        return (
                len( [ i for i in Path(folder).glob('*.gt') ] ),
                len( [ i for i in Path(folder).glob('*.png') ] )
                )
#
