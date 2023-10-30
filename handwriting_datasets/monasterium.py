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
import tarfile

from . import download_utils as du

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2




"""
Generate image/GT pairs for Monasterium transcriptions. This is a multi-use, generic dataset class, that can be used to generate

- line-segmentation dataset
- HTR dataset: Extract line polygons from each image and their respective transcriptions from the PageXML file
- TODO: make downloadable

Directory structure for local storage:

- basefolder : location where is to be downloaded and its content to be extracted (i.e. a subdirectory
  'MonasteriumTekliaGTDataset' containing all image files)
- target_folder: where to create the dataset to be used for given task (segmentation or htr)

"""

logging.basicConfig( level=logging.DEBUG )

logger = logging.getLogger()

class MonasteriumDataset(VisionDataset):

    dataset_file = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
    }

    def __init__(
                self,
                basefolder: str,
                subset: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract_pages: bool = True,
                build_items: bool = True,
                task: str = '',
                target_folder: str ='line_imgs',
                count: int = 0,
                ):
        """
        Args:
            basefolder (str): Where the subfolder containing original files (pageXML documents and page images) 
                           is to be created.
            subset (str): 'train' (default), 'validate' or 'test'.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground truth at loading time.
            extract_pages (bool): if True (default), extract the archive's content into the base folder.
            task (str): 'htr' for HTR set = pairs (line, transcription), 'segm' for segmentation 
                        = cropped TextRegion images, with corresponding PageXML files. Default: ''
            build_items (bool): if True (default), extract and store images for the task from the pages; 
                     otherwise try loading the data from existing, cached data.
            target_folder: Where line images and ground truth transcriptions are created (assumed to 
                           be relative to the caller's pwd).
            count (int): Stops after extracting {count} image items (for testing purpose only).
        """

        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        csv_path = Path(target_folder, 'monasterium_ds.csv')

        super().__init__(basefolder, transform=trf, target_transform=target_transform )

        # the tarball's top folder
        self.setname = 'MonasteriumTekliaGTDataset'
        self.basefolder=basefolder

        basefolder_path = Path( basefolder )
        if not basefolder_path.is_dir():
            print("Cannot extract archive: folder {basefolder} should be created first.", file=sys.stderr)
            sys.exit()

        self.download_and_extract( basefolder_path, basefolder_path, self.dataset_file, extract_pages)

        # input PageXML files are at the root of the resulting tree
        self.pagexmls = Path(basefolder, self.setname).glob('*.xml')

        self.data = []

        # if archive has not been extracted, no point going further
        if not extract_pages or not task:
            return

            return

        if task == 'htr':
            # when DS not created from scratch, try loading from an existing CSV file
            # TO-DO: make generic for both tasks
            if not build_items:
                if csv_path.exists():
                    self.data = self.load_from_csv( csv_path )
            else:
                self.data = self.split_set( self.extract_lines( target_folder, limit=count ), subset )
                # Generate a CSV file with one entry per img/transcription pair
                self.dump_to_csv( csv_path, self.data )
        elif task == 'segm':
            if not build_items:
                pass
            else:
                self.data = self.extract_text_regions( target_folder, limit=count )

#
    def download_and_extract(
            self,
            root: str,
            base_folder_path: Path,
            fl_meta: dict,
            extract=True) -> None:
        """
        Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        TODO: factor out in utility module ??

        Args:
            root: where to save the archive
            base_folder: where to extract (any valid path)
            fl_meta: a dict with file meta-info (keys: url, filename, md5, origin, desc)
        """
        output_file_path = Path(root, fl_meta['filename'])

        #print( fl_meta, type(fl_meta['md5']) )
        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            print("Downloading archive...")
            du.resumable_download(fl_meta['url'], root, fl_meta['filename'], google=(fl_meta['origin']=='google'))

        if not base_folder_path.exists() or not base_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # line images
        if not extract:
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                print('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( base_folder_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                print('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( base_folder_path )




    def purge(self, folder: str) -> int:
        """
        Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the CSV file.

        Args:
            folder (str): Name of the subfolder to purge (relative the caller's
                          pwd
        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
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


    def extract_text_regions(self, target_folder: str, text_only=False, limit=0) -> List[Tuple[str, str]]:
        """
        Crop text regions from original files, and create a new dataset for segmentation where text region image 
        has a corresponding, new PageXML decriptor.

        Args:
            target_folder (str): Line images are extracted in this subfolder (relative to the caller's pwd).
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
            img_path = Path(self.basefolder, self.setname, f'{xml_id}.jpg')
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

                for textregion_elt in page_root.findall( './/pc:TextRegion', ns ):
                    if limit and count == limit:
                        return items

                    textregion = dict()
                    textregion['id']=textregion_elt.get("id")

                    polygon_string=textregion_elt.find('./pc:Coords', ns).get('points')
                    coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                    textregion['bbox'] = IP.Path( coordinates ).getbbox()
                    img_path_prefix = Path(target_folder, f"{xml_id}-{textregion['id']}" )
                    textregion['img_path'] = img_path_prefix.with_suffix('.png')

                    #items.append( (textregion['img_path'], textline['transcription']) ) 
                    if not text_only:
                        bbox_img = page_image.crop( textregion['bbox'] )

                        bbox_img.save( textline['img_path'] )

                    # create a new PageXML file whose a single text region that covers the whole image, 
                    # where line coordinates have been shifted accordingly
                    self.write_region_to_xml( page, ns, textregion )

                    count += 1

        return items


    def write_region_to_xml( self, page, ns, textregion ):
        with open( page, 'r') as page_file:
            page_tree = ET.parse( page_file )
            ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
            page_root = page_tree.getroot()
            page_elt = page_root.find('pc:Page', ns)

            for region_elt in page_elt.findall('pc:TextRegion', ns):
                if region_elt.get('id') != textregion['id']:
                    page_elt.remove( region_elt )
                else:
                    # substract region's coordinates from lines coordinates
                    for line in region_elt.findall('pc:TextLine', ns):
                        coord_elt = line.find('pc:Coords', ns)
                        coordinates =  [ tuple(map(int, pt.split(','))) for pt in coord_elt.get('points').split(' ') ]
                        print( "Old coordinates =", coordinates)
                        print( "text region BBox =", textregion['bbox'] )
                        x_off, y_off = textregion['bbox'][:2]
                        print( "Offset =", x_off, y_off)
                        coordinates = [ (
                                        pt[0]-x_off if pt[0]>x_off else 0, 
                                        pt[1]-y_off if pt[1]>y_off else 0) for pt in coordinates ]
                        print( "New coordinates =", coordinates)
                        transposed_polygon_str = ' '.join([ ','.join([str(p) for p in pt]) for pt in coordinates ] )
                        print("Transposed polygon str =", transposed_polygon_str )
                        coord_elt.set('points', transposed_polygon_str)
                        #sys.exit()


            page_tree.write( textregion['img_path'].with_suffix('.xml') )

        


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
            img_path = Path(self.basefolder, self.setname, f'{xml_id}.jpg')
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
                    #img_sizes.append( [ textline['bbox'][i+2]-textline['bbox'][i] for i in (0,1) ])
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

