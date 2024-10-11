import sys
import logging
import warnings
import random
import tarfile
import hashlib
import json
import shutil
import re
from pathlib import *
from typing import *
from tqdm import tqdm
import defusedxml.ElementTree as ET
#import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from PIL import ImagePath as IP

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms

from . import download_utils as du
from . import xml_utils as xu

from . import alphabet

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
- root/<raw data folder> : where archives are to be extracted (i.e. a subdirectory 'MonasteriumTekliaGTDataset' containing all image files)
- work_folder: where to create the dataset to be used for given task (segmentation or htr)

TODO:

- dataset splitting should return 3 sets together? 
- build_items=False option needlessly complicated: better with 'load_from_tsv_file=<tsv path>' option
  whose directory is assumed to be the work folder

"""

logging.basicConfig( level=logging.DEBUG )

logger = logging.getLogger()

# this is the tarball's top folder, automatically created during the extraction  (not configurable)
tarball_root_name="MonasteriumTekliaGTDataset"    
work_folder_name="MonasteriumHandwritingDataset"
root_folder_basename="Monasterium"
alphabet_tsv_name="monasterium_alphabet.tsv"


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
                root: str='',
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                subset_ratios: Tuple[float,float,float]=(1., 0., 0.),
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract_pages: bool = False,
                from_tsv_file: str = '',
                build_items: bool = True,
                task: str = '',
                shape: str = '',
                count: int = 0,
                alphabet_tsv: str = None,
                ):
        """
        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing original files
                        (pageXML documents and page images) is to be created. Default: 
                        subfolder `data/Monasterium' in this project's directory.
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                               are to be created; default: '<root>/MonasteriumHandwritingDatasetHTR'; if 
                               parameter is a relative path, the work folder is created under <root>; an
                               absolute path overrides this. For HTR task, the work folder also contains
                               the alphabet in TSV form.
            subset (str): 'train' (default), 'validate' or 'test'.
            subset_ratios (Tuple[float, float, float]): ratios for respective ('train', 'validate', ...) subsets
            transform (Callable): Function to apply to the PIL image at loading time.
            target_transform (Callable): Function to apply to the transcription ground truth at loading time.
            extract_pages (bool): if True, extract the archive's content into the base folder no matter what;
                               otherwise (default), check first for a file tree with matching name and checksum.
            task (str): 'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation 
                        = cropped TextRegion images, with corresponding PageXML files. If '' (default),
                        the dataset archive is extracted but no actual data get built.
            shape (str): 'bbox' (default) for line bounding boxes or 'polygons' 
            build_items (bool): if True (default), extract and store images for the task from the pages; 
                     otherwise, just extract the original data from the archive.
            from_tsv_file (str): TSV file from which the data are to be loaded (containing folder is
                                 assumed to be the work folder, superceding the work_folder option).
            count (int): Stops after extracting {count} image items (for testing purpose only).
            alphabet_tsv (str): TSV file containing the alphabet
        """
        
        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        super().__init__(root, transform=trf, target_transform=target_transform )

        self.root = Path(root) if root else Path(__file__).parent.joinpath('data', root_folder_basename)
        print("Root folder: {}".format( self.root ))
        if not self.root.exists():
            self.root.mkdir( parents=True )
            print("Create root path: {}".format(self.root))

        self.work_folder_path = None # task-dependent
        # tarball creates its own base folder
        self.raw_data_folder_path = self.root.joinpath( tarball_root_name )

        if from_tsv_file == '':
            self.download_and_extract( self.root, self.root, self.dataset_file, extract_pages)

        # input PageXML files are at the root of the resulting tree
        # (sorting is necessary for deterministic output)
        self.pagexmls = sorted( Path(self.root, tarball_root_name ).glob('*.xml'))

        self.data = []

        # Used only for HTR tasks: initialized by build_task()
        self.alphabet = None

        self._task = ''
        if (task != ''):
            self._task = task # for self-documentation only
            build_ok = build_items if from_tsv_file == '' else False
            self.build_task( task, build_items=build_ok, from_tsv_file=from_tsv_file, 
                             subset=subset, shape=shape, subset_ratios=subset_ratios, 
                             work_folder=work_folder, count=count, alphabet_tsv=alphabet_tsv )

            #print("\nbuild_task(): data=", self.data[:6])

    def build_task( self, 
                   task: str='htr',
                   build_items: bool=True, 
                   from_tsv_file: str='',
                   subset: str='train', 
                   subset_ratios: Tuple[float,float,float]=(1., 0., 0.),
                   shape: str='bbox', 
                   count: int=0, 
                   work_folder: str='', 
                   crop=False,
                   alphabet_tsv='',
                   ):
        """
        From the read-only, uncompressed archive files, build the image/GT files required for the task at hand.
        + only creates the files needed for a particular task (train, validate, or test): if more than one subset is needed, just initialize a new dataset with desired parameters (work directory, subset)
        + by default, 'train' subset contains 60% of the samples, 'validate', 10%, and 'test', 20%.
        + set samples are randomly picked, but two subsets are guaranteed to be complementary.

        Args:
            subset (str): 'train', 'validate' or 'test'.
            subset_ratios (Tuple[float, float, float]): ratios for respective ('train', 'validate', ...) subsets
            build_items (bool): if True (default), extract and store images for the task from the pages; 
            task (str): 'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation 
            shape (str): 'bbox' (default) for line bounding boxes or 'polygons'
            crop (bool): (for segmentation set only) crop text regions from both image and PageXML file.
            count (int): Stops after extracting {count} image items (for testing purpose only).
            from_tsv_file (str): TSV file from which the data are to be loaded (containing folder is
                                 assumed to be the work folder, superceding the work_folder option).
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                               are to be created; default: './MonasteriumHandwritingDatasetHTR'.
            alphabet_tsv (str): TSV file containing the alphabet
        """
        if task == 'htr':
            
            if crop:
                self.print("Warning: the 'crop' [to WritingArea] option ignored for HTR dataset.")
            
            # when DS not created from scratch, try loading 
            # - from an existing TSV file 
            # - actual folder 
            # - or pickled tensors?
            # TO-DO: make generic for both tasks
            if from_tsv_file != '':
                tsv_path = Path( from_tsv_file )
                if tsv_path.exists():
                    self.work_folder_path = tsv_path.parent
                    # paths are assumed to be absolute
                    self.data = self.load_from_tsv( tsv_path )
                    #print("\nbuild_task(): data=", self.data[:6])
                    #print(self.data[0]['height'], "type=", type(self.data[0]['height']))
                    #print(self.data[0]['polygon_mask'])
            else:
                if work_folder=='':
                    self.work_folder_path = Path(self.root, work_folder_name+'HTR') 
                    print("Setting default location for work folder: {}".format( self.work_folder_path ))
                else:
                    # if work folder is an absolute path, it overrides the root
                    self.work_folder_path = self.root.joinpath( work_folder )
                    print("Work folder: {}".format( self.work_folder_path ))

                if not self.work_folder_path.is_dir():
                    self.work_folder_path.mkdir()
                    print("Creating work folder = {}".format( self.work_folder_path ))

                self.data = self.split_set( self.extract_lines( self.raw_data_folder_path, self.work_folder_path, count=count, shape=shape ), ratios=subset_ratios, subset=subset )
                
                # Generate a TSV file with one entry per img/transcription pair
                self.dump_data_to_tsv( Path(self.work_folder_path.joinpath(f"monasterium_ds_{subset}.tsv")) )
                self.generate_readme("README.md", 
                                     {'subset':subset, 'task':task, 'shape':shape, 'count':count, 'work_folder': work_folder })

                # copy the alphabet into the work folder
                shutil.copy(self.root.joinpath( alphabet_tsv_name ), self.work_folder_path )
            
            # load alphabet
            alphabet_tsv_input = Path( alphabet_tsv ) if alphabet_tsv else self.work_folder_path.joinpath( alphabet_tsv_name )
            if not alphabet_tsv_input.exists():
                raise FileNotFoundError("Alphabet file: {}".format( alphabet_tsv_input))
            print('alphabet path:', alphabet_tsv_input)
            self.alphabet = alphabet.Alphabet( alphabet_tsv_input )
                

        elif task == 'segment':
            self.work_folder_path = Path('.', work_folder_name+'Segment') if work_folder=='' else Path( work_folder )
            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir() 

            if build_items:
                if crop:
                    self.data = self.extract_text_regions( self.raw_data_folder_path, self.work_folder_path, count=count )
                else:
                    self.data = self.build_page_lines_pairs( self.raw_data_folder_path, self.work_folder_path, count=count )


    def generate_readme( self, filename: str, params: dict ):
        """
        Create a metadata file in the work directory.
        """
        filepath = Path(self.work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n' + 
                  '\n\t+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)

    def get_paths( self ):
        return """
        Archive: {}
        Expanded archive: {}
        Current task folder: {}
        """.format( 
                Path(self.root, self.dataset_file['filename']), 
                self.raw_data_folder_path, 
                self.work_folder_path if self.work_folder_path else '(no task defined)')

#
    def download_and_extract(
            self,
            root: Path,
            raw_data_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """
        Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root: where to save the archive
            raw_data_folder: where to extract (any valid path)
            fl_meta: a dict with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
        """
        output_file_path = root.joinpath( fl_meta['filename'])

        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            print("Downloading archive...")
            du.resumable_download(fl_meta['url'], root, fl_meta['filename'], google=(fl_meta['origin']=='google'))
        else:
            print("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_file['md5']))

        if not raw_data_folder_path.exists() or not raw_data_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract and du.check_extracted( raw_data_folder_path.joinpath( tarball_root_name ) , fl_meta['full-md5'] ):
            print('Found valid file tree in {}: skipping the extraction stage.'.format(str(raw_data_folder_path.joinpath( tarball_root_name ))))
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                print('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                print('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )


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


    def dump_data_to_tsv(self, file_path: str='', all_path_style=False):
        """
        Create a CSV file with all tuples (<line image absolute path>, <transcription>, <original height>, <original width> [<polygon points]).

        Args:
            file_path (str): A TSV (absolute) file path 
            all_path_style (bool): list GT file name instead of GT content.
        """
        if file_path == '':
            for sample in self.data:
                # note: TSV only contains the image file name (load_from_tsv() takes care of applying the correct path prefix)
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                print("{}\t{}\t{}\t{}".format( img_path, 
                      gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'), int(height), int(width)))
            return
        with open( file_path, 'w' ) as of:
            for sample in self.data:
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                #print('{}\t{}'.format( img_path, gt, height, width ))
                of.write( '{}\t{}\t{}\t{}'.format( img_path,
                                             gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'),
                                             int(height), int(width) ))
                if 'polygon_mask' in sample and sample['polygon_mask'] is not None:
                    of.write('\t{}'.format( sample['polygon_mask'] ))
                of.write('\n')
                                            

    @staticmethod
    def load_from_tsv(file_path: Path) -> List[dict]:
        """
        Load samples (as dictionaries) from an existing TSV file.

        Args:
            file_path (Path): A file path (relative to the caller's pwd), where each line
                                      is either a tuple 

            ```python
            <img file path> <transcription text> <height> <width> [<polygon points>]
            ````
            or

            ```python
            <img file path> <transcription file path> <height> <width> [<polygon points>]
            ````

        Returns:
            List[dict]: A list of dictionaries of the form {'img': <img file path>,
                                                      'transcription': <transcription text>,
                                                      'height': <original height>,
                                                      'width': <original width>,
                                                      'mask': <mask for unpadded part of the img>}}
        """
        work_folder_path = file_path.parent
        samples=[]
        with open( file_path, 'r') as infile:
            # Detection: 
            # - is the transcription passed as a filepath or as text?
            # - is there a polygon path

            #img_path, file_or_text, height, width, polygon_mask = [None] * 5
            has_polygon = False
            fields = next( infile )[:-1].split('\t')
            img_path, file_or_text, height, width = fields[:4]
            polygon_mask = None
            if len(fields) > 4:
                polygon_mask = fields[4]
                has_polygon = True
            print('load_from_tsv(): type(img_path)=', type(img_path), 'type(height)=', type(height))
            all_path_style = True if Path(file_or_text).exists() else False
            infile.seek(0) 
            if not all_path_style:
                def tsv_to_dict( tsv_line ):
                    img, transcription, height, width, polygon_mask = [ None ] * 5
                    if has_polygon:
                        img, transcription, height, width, polygon_mask = tsv_line[:-1].split('\t')
                        #print('tsv_to_dict(): type(height)=', type(height))

                        s = {'img': str(work_folder_path.joinpath( img )), 'transcription': transcription,
                                'height': int(height), 'width': int(width), 'polygon_mask': eval(polygon_mask) }
                        #print('tsv_to_dict(): type(img_path)=', type(s['img']), 'type(s[height]=', type(s['height']))
                        return s
                    else:
                        img, transcription, height, width = tsv_line[:-1].split('\t')
                        s = {'img': str(work_folder_path.joinpath( img )), 'transcription': transcription,
                                'height': int(height), 'width': int(width) }
                        #print('tsv_to_dict(): type(img_path)=', type(s['img']), 'type(s[height]=', type(s['height']))
                        return s

                samples = [ tsv_to_dict(s) for s in infile ]
                #print("tsv_to_dict(): samples=", samples)
            else:
                for tsv_line in infile:
                    img_file, gt_file, height, width = tsv_line[:-1].split('\t')
                    with open( work_folder_path.joinpath( gt_file ), 'r') as igt:
                        gt_content = '\n'.join( igt.readlines())
                        if has_polygon:
                            samples.append( {'img': str(work_folder_path.joinpath( img_file )), 'transcription': gt_content,
                                             'height': int(height), 'width': int(width), 'polygon_mask': eval(polygon_mask) })
                        else:
                            samples.append( {'img': str(work_folder_path.joinpath( img_file )), 'transcription': gt_content,
                                             'height': int(height), 'width': int(width) })

        return samples


    def build_page_lines_pairs(self, raw_data_folder_path:Path, work_folder_path: Path, text_only:bool=False, count:int=0, metadata_format:str='xml') -> List[Tuple[str, str]]:
        """
        Create a new dataset for segmentation that associate each page image with its metadata.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            count (int): Stops after extracting {count} images (for testing purpose).
            metadata_format (str): 'xml' (default) or 'json'

        Returns:
            list: a list of pairs (<absolute img filepath>, <absolute transcription filepath>)
        """
        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        items = []

        for page in self.pagexmls:
            xml_id = Path( page ).stem
            img_path = Path(self.raw_data_folder_path, f'{xml_id}.jpg')

            img_path_dest = work_folder_path.joinpath( img_path.name )
            xml_path_dest = work_folder_path.joinpath( page.name )

            # copy both image and xml file into work directory
            shutil.copy( img_path, img_path_dest )
            shutil.copy( page, xml_path_dest )
            items.append( img_path_dest, xml_path_dest )

        return items


    def extract_text_regions(self, raw_data_folder_path: Path, work_folder_path: Path, text_only=False, count=0, metadata_format:str='xml') -> List[Tuple[str, str]]:
        """
        Crop text regions from original files, and create a new dataset for segmentation where the text
        region image has a corresponding, new PageXML decriptor.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            count (int): Stops after extracting {count} images (for testing purpose).
            metadata_format (str): 'xml' (default) or 'json'
/
        Returns:
            list: a list of pairs (img_file_path, transcription)
        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        cnt = 0 # for testing purpose

        items = []

        for page in self.pagexmls:

            xml_id = Path( page ).stem
            img_path = Path(self.raw_data_folder_path, f'{xml_id}.jpg')

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
                    if cnt and cnt == count:
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

                    cnt += 1

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


    def extract_lines(self, raw_data_folder_path: Path, work_folder_path: Path, shape='bbox', text_only=False, count=0) -> List[Dict[str, Union[Tensor,str,int]]]:
        """
        Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            shape (str): Extract lines as bboxes (default) or as polygon-within-bbox.
            text_only (bool): Store only the transcriptions (*.gt.txt files).
            count (int): Stops after extracting {count} images (for testing purpose).

        Returns:
            list[dict]: An array of dictionaries {'img': <absolute img_file_path>,
                                                  'transcription': <transcription text>,
                                                  'height': <original height>,
                                                  'width': <original width>}

        """
        print("extract_lines()")
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        gt_lengths = []
        img_sizes = []
        cnt = 0 # for testing purpose

        samples = [] # each sample is a dictionary {'img': <img file path> , 'transcription': str,
                     #                              'height': int, 'width': int}

        for page in tqdm(self.pagexmls):

            xml_id = Path( page ).stem
            img_path = Path( raw_data_folder_path, f'{xml_id}.jpg')
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
                    if count and cnt == count:
                        return samples

                    textline = dict()
                    textline['id']=textline_elt.get("id")
                    textline['transcription']=textline_elt.find('./pc:TextEquiv', ns).find('./pc:Unicode', ns).text

                    # skip lines that don't have a transcription
                    if not textline['transcription']:
                        continue

                    polygon_string=textline_elt.find('./pc:Coords', ns).get('points')
                    coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                    textline['bbox'] = IP.Path( coordinates ).getbbox()
                    
                    x_left, y_up, x_right, y_low = textline['bbox']
                    textline['width'], textline['height'] = x_right-x_left, y_low-y_up
                    
                    img_path_prefix = work_folder_path.joinpath( f"{xml_id}-{textline['id']}" )
                    textline['img_path'] = img_path_prefix.with_suffix('.png')
                    textline['polygon'] = None

                    #print("extract_lines():", samples[-1])
                    if not text_only:
                        bbox_img = page_image.crop( textline['bbox'] )

                        if shape=='bbox':
                            bbox_img.save( textline['img_path'] )

                        else:
                            # This takes care of saving line images, but not of computing
                            # useful information for training and inference, such as masks.
                            #
                            # Options:
                            # 1. Compute mask at this stage and add it to the sample: added complexity
                            # 2. Compute mask in __getitem()__: consistent with having all tensor processing
                            #    at sample use stage, but need at least polygon definition
                            # 3. Find way to save mask in TSV? Related to...
                            # 4. Add another import/export format (pickled tensors?)

                            # Solution = everything :))
                            # 1. Add polygon sequence to raw sample
                            # 2. Serialize and deserialize same seq. to/fro TSV
                            # 3. Use __getitem__ to compute the tensor from it
                            # 4. Ideally: import/export to pickled tensors

                            # mask (=line polygon)
                            mask = Image.new("L", bbox_img.size, 0)
                            drawer = ImageDraw.Draw(mask)
                            leftx, topy = textline['bbox'][:2]
                            transposed_coordinates = [ (x-leftx, y-topy) for x,y in coordinates ]
                            drawer.polygon( transposed_coordinates, fill=255 )
                            textline['polygon']=transposed_coordinates

                            # background
                            bg = Image.new("RGB", bbox_img.size, color=0)

                            # composite
                            Image.composite(bbox_img, bg, mask).save( Path( textline['img_path'] ))


                    sample = {'img': textline['img_path'], 'transcription': textline['transcription'], \
                               'height': textline['height'], 'width': textline['width'] }
                    #print("extract_lines(): sample=", sample)
                    if textline['polygon'] is not None:
                        sample['polygon_mask'] = textline['polygon']
                    samples.append( sample )

                    with open( img_path_prefix.with_suffix('.gt.txt'), 'w') as gt_file:
                        gt_file.write( textline['transcription'] )
                        gt_lengths.append(len( textline['transcription']))

                    cnt += 1

        return samples
    


    @staticmethod
    def split_set(samples: object, ratios: Tuple[float, float, float], subset) -> List[object]:
        """
        Split a dataset into 3 sets: train, validation, test.

        Args:
            samples (object): any dataset sample.
            ratios Tuple[float, float, float]: respective proportions for possible subsets
            subset (str): subset to be build  ('train', 'validate', or 'test')

        Returns:
            list[object]: a list of samples.
        """

        random.seed(10)

        if 1.0 in ratios:
            return list( samples )

        subset_2_count = int( len(samples)* ratios[1])
        subset_3_count = int( len(samples)* ratios[2] )

        subset_1_indices = set( range(len(samples)))
        
        if ratios[1] != 0:
            subset_2_indices = set( random.sample( subset_1_indices, subset_2_count))
            subset_1_indices -= subset_2_indices

        if ratios[2] != 0:
            subset_3_indices = set( random.sample( subset_1_indices, subset_3_count))
            subset_1_indices -= subset_3_indices

        if subset == 'train':
            return [ samples[i] for i in subset_1_indices ]
        if subset == 'validate':
            return [ samples[i] for i in subset_2_indices ]
        if subset == 'test':
            return [ samples[i] for i in subset_3_indices ]


    def __getitem__(self, index) -> dict:
        """
        Returns:
            dict[str,Union[Tensor,int,str]]: dictionary
        """
        img_path, height, width, gt = self.data[index]['img'], self.data[index]['height'],\
                                                 self.data[index]['width'], self.data[index]['transcription']
        #print('__getitem__(): data[{}]={}'.format(index, self.data[index]))
        #polygon_mask = self.data[index]['polygon_mask'] if 'polygon_mask' in self.data[index] else None

        assert isinstance(img_path, Path) or isinstance(img_path, str)
        assert type(height) is int
        assert type(width) is int
        assert type(gt) is str
        assert type(gt_len) is int

        # goal: transform image, while returning not only the image but also the unpadded size
        # -> meta-information has to be passed along in the sample; :
        sample_with_img_file = self.data[index].copy()
        sample_with_img_file['img'] = Image.open( sample_with_img_file['img'], 'r')
        #print('__getitem__({}): sample='.format(index), sample_with_img_file)
        return self.transform( sample_with_img_file )


    def __len__(self) -> int:
        """
        Returns:
            int: number of data points.
        """
        return len( self.data )

    @property
    def task( self ):
        if self._task == 'htr':
            return "HTR"
        if self._task == 'segment':
            return "Segmentation"
        return "None defined."

    def __str__(self) -> str:
        return (f"Root folder:\t{self.root}\n"
               f"Files extracted in:\t{self.root.joinpath(tarball_root_name)}\n"
               f"Task: {self.task}\n"
               f"Work folder:\t{self.work_folder_path}\n"
               f"Data points:\t{len(self.data)}")


    def count_line_items(self, folder) -> Tuple[int, int]:
        return (
                len( [ i for i in Path(folder).glob('*.gt.txt') ] ),
                len( [ i for i in Path(folder).glob('*.png') ] )
                )


class ResizeToMax():

    def __init__( self, max_h, max_w ):
        self.max_h, self.max_w = max_h, max_w

    def __call__(self, sample):
        t, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        if h <= self.max_h and w <= self.max_w:
            return sample
        t = v2.Resize(size=self.max_h, max_size=self.max_w, antialias=True)( t )
        h_new, w_new = [ int(d) for d in t.shape[1:] ]
        
        return {'img': t, 'height': h_new, 'width': w_new, 'transcription': gt }

class PadToHeight():

    def __init__( self, max_h ):
        self.max_h = max_h

    def __call__(self, sample):
        t, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        if h > self.max_h:
            warnings.warn("Cannot pad an image that is higher ({}) than the padding size ({})".format( h, self.max_h))
            return sample
        new_t = torch.zeros( (t.shape[0], self.max_h, t.shape[2]))
        new_t[:,:h,:] = t

        # add a field
        mask = torch.zeros( new_t.shape, dtype=torch.bool)
        mask[:,:h,:]=1
        return {'img': new_t, 'height': h, 'width': w, 'transcription': gt, 'mask': mask }

class PadToWidth():

    def __init__( self, max_w ):
        self.max_w = max_w

    def __call__(self, sample):
        t_chw, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        if w > self.max_w:
            warnings.warn("Cannot pad an image that is wider ({}) than the padding size ({})".format( w, self.max_w))
            return sample
        new_t_chw = torch.zeros( t_chw.shape[:2] + (self.max_w,))
        new_t_chw[:,:,:w] = t_chw

        # add a field
        mask = torch.zeros( new_t_chw.shape, dtype=torch.bool)
        mask[:,:,:w] = 1
        return {'img': new_t_chw, 'height': h, 'width': w, 'transcription': gt, 'mask': mask }


class PadToSize():

    def __init__( self, max_h, max_w ):
        self.max_h, self.max_w = max_h, max_w

    def __call__( self, sample ):
        t, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        if h > self.max_h or w > self.max_w:
            warnings.warn("Cannot pad an image that is larger ({}x{}) than the padding size ({}x{})".format(
                h, w, self.max_h, self.max_w))
            return sample
        new_t = torch.zeros( (t.shape[0], self.max_h, self.max_w) )
        new_t[:,:h,:w]=t

        # add a field
        mask = torch.zeros( new_t.shape, dtype=torch.bool)
        mask[:,:h,:w]=1
        return {'img': new_t, 'height': h, 'width': w, 'transcription': gt, 'mask': mask }

# unused for the moment
class ResizeToMaxHeight():

    def __init__( self, max_h ):
        self.max_h = max_h

    def __call__(self, sample):
        t, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        if h <= self.max_h:
            return sample
        # freak case (marginal annotations): original height is the larger
        # dimension -> specify the width too
        if h > w and h > max_h:
            t = v2.Resize( size=(self.max_h, int(w*self.max_h/h) ), antialias=True)(t)
        else:
        # default case: original height is the smaller dimension and 
        # gets picked up by Resize()
            t = v2.Resize(size=self.max_h, antialias=True)( t )
        h_new, w_new = [ int(d) for d in t.shape[1:] ]

        return {'img': t, 'height': h_new, 'width': w_new, 'transcription': gt }

class ResizeToHeight():
    """
    Resize an image with fixed height, preserving aspect ratio as long as the resulting width does
    not exceed the specified max. width. If that is the case, the image is horizontally squeezed 
    to fix this.
    """

    def __init__( self, target_height, max_width ):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, sample):
        t_chw, h, w, gt = sample['img'], sample['height'], sample['width'], sample['transcription']
        # freak case (marginal annotations): original height is the larger
        # dimension -> specify the width too
        if h > w:
            t_chw = v2.Resize( size=(self.target_height, int(w*self.target_height/h) ), antialias=True)( t_chw )
        # default case: original height is the smaller dimension and
        # gets picked up by Resize()
        else:
            t_chw = v2.Resize(size=self.target_height, antialias=True)( t_chw )
            
        if t_chw.shape[-1] > self.max_width:
            t_chw = v2.Resize(size=(self.target_height, self.max_width), antialias=True)( t_chw )
        h_new, w_new = [ int(d) for d in t_chw.shape[1:] ]

        mask = torch.zeros( t_chw.shape, dtype=torch.bool)
        mask[:,:h,:w]=1
        return {'img': t_chw, 'height': h_new, 'width': w_new, 'transcription': gt, 'mask': mask }

# check that the module is testable
def dummy():
    return True


def dummy():
    return True
