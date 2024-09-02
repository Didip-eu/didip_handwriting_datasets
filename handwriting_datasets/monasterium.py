import sys
from typing import *
import defusedxml.ElementTree as ET
#import xml.etree.ElementTree as ET
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
import subprocess
import hashlib
import json
import shutil

from . import download_utils as du
from . import xml_utils as xu

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2

class DataException( Exception ):
    pass


"""
Generate image/GT pairs for Monasterium transcriptions. This is a multi-use, generic dataset class, that can be used to generate

- line-segmentation dataset
- HTR dataset: Extract line polygons from each image and their respective transcriptions from the PageXML file


Directory structure for local file storage:

- root: where datasets archives are to be downloaded
- root/<base_folder> : where archives are to be extracted (i.e. a subdirectory 'MonasteriumTekliaGTDataset' containing all image files)
- work_folder: where to create the dataset to be used for given task (segmentation or htr)

"""

logging.basicConfig( level=logging.DEBUG )

logger = logging.getLogger()

# this is the tarball's top folder, automatically created during the extraction  (not configurable)
tarball_root_name="MonasteriumTekliaGTDataset"    
work_folder_name="MonasteriumHandwritingDataset"


class MonasteriumDataset(VisionDataset):

    dataset_file = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
    }

    def __init__(
                self,
                root: str=str(Path.home().joinpath('tmp', 'data', 'Monasterium')),
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract_pages: bool = False,
                build_items: bool = True,
                task: str = '',
                shape: str = '',
                count: int = 0,
                ):
        """
        Args:
            root (str): Where the subfolder containing original files (pageXML documents and page images) 
                        is to be created.
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                               are to be created; default: './MonasteriumHandwritingDatasetHTR'.
            subset (str): 'train' (default), 'validate' or 'test'.
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground truth at loading time.
            extract_pages (bool): if True, extract the archive's content into the base folder no matter what;
                               otherwise (default), check first for a file tree with matching name and checksum.
            task (str): 'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation 
                        = cropped TextRegion images, with corresponding PageXML files. If '' (default),
                        the dataset archive is extracted but no actual data get built.
            shape (str): 'bbox' for line bounding boxes or 'polygons' (default)
            build_items (bool): if True (default), extract and store images for the task from the pages; 
                     otherwise try loading the data from existing, cached data (in which case, work_folder
                     option must be non-empty).
            count (int): Stops after extracting {count} image items (for testing purpose only).
        """

        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        super().__init__(root, transform=trf, target_transform=target_transform )

        self.root = root
        self.work_folder_path = None # task-dependent
        # tarball creates its own base folder
        self.base_folder_path = Path( self.root, tarball_root_name )
        self.download_and_extract( root, Path(root), self.dataset_file, extract_pages)

        # input PageXML files are at the root of the resulting tree
        self.pagexmls = Path(root, tarball_root_name ).glob('*.xml')

        self.data = []

        print(self.get_paths())

        if (task != ''):
            self.build_task( task, build_items=build_items, subset=subset, shape=shape, work_folder=work_folder )


    def build_task( self, task: str='htr', build_items: bool=True, subset: str='train', shape: str='polygons', count: int=0, work_folder: str='', crop=False):
        """
        From the read-only, uncompressed archive files, build the image/GT files required for the task at hand.

        Args:
            subset (str): 'train' (default), 'validate' or 'test'.
            build_items (bool): if True (default), extract and store images for the task from the pages; 
            task (str): 'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation 
            shape (str): 'bbox' for line bounding boxes or 'polygons' (default)
            crop (bool): (for segmentation set only) crop text regions from both image and PageXML file.
            count (int): Stops after extracting {count} image items (for testing purpose only).
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                               are to be created; default: './MonasteriumHandwritingDatasetHTR'.
        """

        if task == 'htr':
            print('work_folder=' + work_folder )
            if crop:
                self.print("Warning: the 'crop' [to WritingArea] option ignored for HTR dataset.")
            self.work_folder_path = Path('.', work_folder_name+'HTR') if work_folder=='' else Path( work_folder )
            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir() 

            tsv_path = self.work_folder_path.joinpath('monasterium_ds.tsv')
            # when DS not created from scratch, try loading 
            # - from an existing TSV file 
            # - actual folder 
            # - or pickled tensors?
            # TO-DO: make generic for both tasks
            if not build_items:
                tsv_path = self.work_folder_path.joinpath('monasterium_ds.tsv')
                if work_folder == '':
                    print("Pass a -work_folder option to specify the data folder")
                if tsv_path.exists():
                    self.data = self.load_from_tsv( tsv_path )
            else:
                self.data = self.split_set( self.extract_lines( self.base_folder_path, self.work_folder_path, limit=count, shape=shape ), subset )
                # Generate a TSV file with one entry per img/transcription pair
                self.dump_data_to_tsv( tsv_path )
        elif task == 'segment':
            self.work_folder_path = Path('.', work_folder_name+'Segment') if work_folder=='' else Path( work_folder )
            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir() 
            if not build_items:
                pass
            elif crop:
                self.data = self.extract_text_regions( self.base_folder_path, self.work_folder_path, limit=count )
            else:
                self.data = self.build_page_lines_pairs( self.base_folder_path, self.work_folder_path, limit=count )



    def get_paths( self ):
        return """
        Archive: {}
        Expanded archive: {}
        Current task folder: {}
        """.format( 
                Path(self.root, self.dataset_file['filename']), 
                self.base_folder_path, 
                self.work_folder_path if self.work_folder_path else '(no task defined)')

#
    def download_and_extract(
            self,
            root: str,
            base_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """
        Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root: where to save the archive
            base_folder: where to extract (any valid path)
            fl_meta: a dict with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
        """
        output_file_path = Path(root, fl_meta['filename'])

        #print( fl_meta, type(fl_meta['md5']) )
        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            print("Downloading archive...")
            du.resumable_download(fl_meta['url'], root, fl_meta['filename'], google=(fl_meta['origin']=='google'))
        else:
            print("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_file['md5']))

        if not base_folder_path.exists() or not base_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract and du.check_extracted( base_folder_path.joinpath( tarball_root_name ) , fl_meta['full-md5'] ):
            print('Found valid file tree in {}: skipping the extraction stage.'.format(str(base_folder_path.joinpath( tarball_root_name ))))
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
        deleted, as well as the TSV file.

        Args:
            folder (str): Name of the subfolder to purge (relative the caller's
                          pwd
        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
            item.unlink()
            cnt += 1
        return cnt


    def dump_data_to_tsv(self, file_path: str=''):
        """
        Create a CSV file with all pairs (<line image absolute path>, <transcription>).

        Args:
            file_path (str): A TSV file path (relative to the caller's pwd).
        """
        if file_path == '':
            for path, gt in self.data:
                print("{}\t{}".format( Path(path).absolute(), Path(gt).absolute()))
            return
        with open( file_path, 'w' ) as of:
            for path, gt in self.data:
                #print('{}\t{}'.format( path, gt ))
                of.write( '{}\t{}\n'.format( Path(path).absolute(), Path(gt).absolute() ) )


    def load_from_tsv(self, file_path: Path) -> list:
        """
        Load pairs (<line image path>, transcription) from an existing CSV file.

        Args:
            file_path (pathlib.Path): A file path (relative to the caller's pwd).

        Returns:
            list: A list of 2-tuples whose first element is the (relative) path of
                  the line image and the second element is the transcription file (*.pt).
        """
        with open( file_path, 'r') as infile:
            return [ pair[:-1].split('\t') for pair in infile ]




    def build_page_lines_pairs(self, base_folder_path:Path, work_folder_path: Path, text_only:bool=False, limit:int=0, metadata_format:str='xml') -> List[Tuple[str, str]]:
        """
        Create a new dataset for segmentation that associate each page image with its metadata.

        Args:
            base_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            limit (int): Stops after extracting {limit} images (for testing purpose).
            metadata_format (str): 'xml' (default) or 'json'

        Returns:
            list: a list of pairs (img_file_path, transcription)
        """
        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        items = []

        for page in self.pagexmls:
            xml_id = Path( page ).stem
            img_path = Path(self.base_folder_path, f'{xml_id}.jpg')

            img_path_dest = work_folder_path.joinpath( img_path.name )
            xml_path_dest = work_folder_path.joinpath( page.name )

            # copy both image and xml file into work directory
            shutil.copy( img_path, img_path_dest )
            shutil.copy( page, xml_path_dest )
            items.append( img_path_dest, xml_path_dest )

        return items


    def extract_text_regions(self, base_folder_path: Path, work_folder_path: Path, text_only=False, limit=0, metadata_format:str='xml') -> List[Tuple[str, str]]:
        """
        Crop text regions from original files, and create a new dataset for segmentation where the text
        region image has a corresponding, new PageXML decriptor.

        Args:
            base_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            limit (int): Stops after extracting {limit} images (for testing purpose).
            metadata_format (str): 'xml' (default) or 'json'
/
        Returns:
            list: a list of pairs (img_file_path, transcription)
        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        count = 0 # for testing purpose

        items = []

        for page in self.pagexmls:

            xml_id = Path( page ).stem
            img_path = Path(self.base_folder_path, f'{xml_id}.jpg')

            with open(page, 'r') as page_file:

                page_tree = ET.parse( page_file )

                ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                        'xsi': "http://www.w3.org/2001/XMLSchema-instance" }

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

                    # Do not use: nominal boundaries of the text regions do not contain
                    # all baseline points
                    #
                    #polygon_string=textregion_elt.find('pc:Coords', ns).get('points')
                    #coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                    #textregion['bbox'] = IP.Path( coordinates ).getbbox()

                    # fix bbox for given region, according to the line points it contains
                    textregion['bbox'] = self.compute_bbox( page, textregion['id'] )
                    if textregion['bbox'] == (0,0,0,0):
                        continue
                    img_path_prefix = work_folder_path.joinpath( f"{xml_id}-{textregion['id']}" )
                    textregion['img_path'] = img_path_prefix.with_suffix('.png')
                    print('textregion["img_path"] =', textregion['img_path'], "type =", type(textregion['img_path']))
                    textregion['size'] = [ textregion['bbox'][i+2]-textregion['bbox'][i]+1 for i in (0,1) ]

                    if not text_only:
                        bbox_img = page_image.crop( textregion['bbox'] )
                        bbox_img.save( textregion['img_path'] )

                    # create a new descriptor file whose a single text region that covers the whole image, 
                    # where line coordinates have been shifted accordingly
                    self.write_region_to_xml( page, ns, textregion )

                    count += 1

        return items


    def compute_bbox(self, page: str, region_id: str ) -> Tuple[int, int, int, int]:
        """
        In the raw Monasterium/Teklia PageXMl file, baseline and/or textline polygon points
        may be outside the nominal boundaries of their text region. This method computes a
        new bounding box for the given text region, based on the baseline points its contains.

        Args:
            page (str): path to the PageXML file.
            region_id (str): id attribute of the region element in the PageXML file

        Returns:
            tuple: region's new coordinates, as (x1, y1, x2, y2)
        """

        def within(pt, bbox):
            return pt[0] >= bbox[0] and pt[0] <= bbox[2] and pt[1] >= bbox[1] and pt[1] <= bbox[3]

        with open(page, 'r') as page_file:

            page_tree = ET.parse( page_file )
            ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
            page_root = page_tree.getroot()

            region_elt = page_root.find( ".//pc:TextRegion[@id='{}']".format( region_id), ns )
            polygon_string=region_elt.find('pc:Coords', ns).get('points')
            coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
            original_bbox = IP.Path( coordinates ).getbbox()

            all_points = []
            valid_lines = 0
            for line_elt in region_elt.findall('pc:TextLine', ns):
                for elt_name in ['pc:Baseline', 'pc:Coords']:
                    elt = line_elt.find( elt_name, ns )
                    if elt_name == 'pc:Baseline' and elt is None:
                        print('Page {}, region {}: could not find element {} for line {}'.format(
                            page, region_elt.get('id'), elt_name, line_elt.get('id')))
                        continue
                    valid_lines += 1
                    #print( elt.get('points').split(','))
                    all_points.extend( [ tuple(map(int, pt.split(','))) for pt in elt.get('points').split(' ') ])
            if not valid_lines:
                return (0,0,0,0)
            not_ok = [ p for p in all_points if not within( p, original_bbox) ]
            if not_ok:
                print("File {}: invalid points for textregion {}: {} -> extending bbox accordingly".format(page, region_id, not_ok))
            bbox = IP.Path( all_points ).getbbox()
            print("region {}, bbox={}".format( region_id, bbox))
            return bbox


    def write_region_to_xml( self, page: str, ns, textregion: dict ):
        """
        From the given text region data, generates a new PageXML file.

        TODO: fix bug in ImageFilename attribute E.g. NA-RM_14240728_2469_r-r1..jpg
        """

        ET.register_namespace('', ns['pc'])
        ET.register_namespace('xsi', ns['xsi'])

        with open( page, 'r') as page_file:

            page_tree = ET.parse( page_file )
            page_root = page_tree.getroot()
            page_elt = page_root.find('pc:Page', ns)

            # updating imageFilename attribute with region id
            page_elt.set( 'imageFilename', str(textregion['img_path'].name) )

            for region_elt in page_elt.findall('pc:TextRegion', ns):
                if region_elt.get('id') != textregion['id']:
                    page_elt.remove( region_elt )
                else:
                    # Shift text region's new coordinates (they now cover the whole image)
                    coord_elt = region_elt.find('pc:Coords', ns)
                    rg_bbox_str = '0,0 {:.0f},{:.0f}'.format( textregion['size'][0], textregion['size'][1] )
                    coord_elt.set( 'points', rg_bbox_str )
                    # substract region's coordinates from lines coordinates
                    for line_elt in region_elt.findall('pc:TextLine', ns):
                        for elt_name in ['pc:Coords', 'pc:Baseline']:
                            elt = line_elt.find( elt_name, ns)
                            if elt is None:
                                continue
                            points =  [ tuple(map(int, pt.split(','))) for pt in elt.get('points').split(' ') ]
                            x_off, y_off = textregion['bbox'][:2]
                            points = [ (
                                        pt[0]-x_off if pt[0]>x_off else 0, 
                                        pt[1]-y_off if pt[1]>y_off else 0) for pt in points ]
                            transposed_point_str = ' '.join([ ','.join(['{:.0f}'.format(p) for p in pt]) for pt in points ] )
                            elt.set('points', transposed_point_str)

            with open( textregion['img_path'].with_suffix('.xml'), 'bw') as f:
                page_tree.write( f, method='xml', xml_declaration=True, encoding="utf-8" )


    def extract_lines(self, base_folder_path: Path,  work_folder_path: Path, shape='polygons', text_only=False, limit=0) -> List[Tuple[str, str]]:
        """
        Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        Args:
            base_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            shape (str): Extract lines as polygon-within-bbox (default) or as bboxes ('bbox').
            text_only (bool): Store only the transcriptions (*.gt files).
            limit (int): Stops after extracting {limit} images (for testing purpose).

        Returns:
            list: An array of pairs (img_file_path, transcription)

        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        gt_lengths = []
        img_sizes = []
        count = 0 # for testing purpose

        items = []

        for page in self.pagexmls:

            xml_id = Path( page ).stem
            img_path = Path( base_folder_path, f'{xml_id}.jpg')
            #print( img_path )

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
                    
                    img_path_prefix = work_folder_path.joinpath( f"{xml_id}-{textline['id']}" )
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

