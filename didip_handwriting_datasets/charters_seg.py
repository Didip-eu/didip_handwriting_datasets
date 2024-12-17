
# stdlib
import sys
import warnings
import random
import tarfile
import json
import shutil
import re
import os
from pathlib import *
from typing import *

# 3rd-party
from tqdm import tqdm
import defusedxml.ElementTree as ET
#import xml.etree.ElementTree as ET
from PIL import Image, ImagePath
import gzip

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset

from . import download_utils as du


class DataException( Exception ):
    """"""
    pass


"""
Utility classes to manage charter data.

"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)



class ChartersDataset(VisionDataset):
    """A generic dataset class for charters, equipped with a rich set of methods for segmentation tasks:

        * region extraction methods (from original page images and XML metadata)

        Attributes:
            dataset_resource (dict): meta-data (URL, archive name, type of repository).

            work_folder_name (str): The work folder is where a task-specific instance of the data is created; if it not
                passed to the constructor, a default path is constructed, using this default name.

            root_folder_basename (str): A basename for the root folder, that contains
                * the archive
                * the subfolder that is created from it.
                * work folders for specific tasks
                By default, a folder named ``data/<root_folder_basename>`` is created in the *project directory*,
                if no other path is passed to the constructor.

    """

    dataset_resource = None

    work_folder_name = "ChartersHandwritingDataset"

    root_folder_basename="Charters"

    def __init__( self,
                root: str='',
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                extract_pages: bool = False,
                from_line_tsv_file: str = '',
                from_page_xml_dir: str = '',
                from_work_folder: str = '',
                build_items: bool = True,
                shape: str = 'polygon',
                count: int = 0,
                line_padding_style: str = 'median',
                resume_task: bool = False
                ) -> None:
        """Initialize a dataset instance.

        Args:
            root (str): Where the archive is to be downloaded and the subfolder containing
                original files (pageXML documents and page images) is to be created. 
                Default: subfolder `data/Charters' in this project's directory.
            work_folder (str): Where line images and ground truth transcriptions fitting a
                particular task are to be created; default: '<root>/ChartersHandwritingDatasetSegment';
                if parameter is a relative path, the work folder is created under
                <root>; an absolute path overrides this.
            subset (str): 'train' (default), 'validate' or 'test'.
            subset_ratios (Tuple[float, float, float]): ratios for respective ('train', 
                'validate', ...) subsets
            target_transform (Callable): Function to apply to the transcription ground
                truth at loading time.
            extract_pages (bool): if True, extract the archive's content into the base
                folder no matter what; otherwise (default), check first for a file tree 
                with matching name and checksum.
            shape (str): Extract each line as bbox ('bbox': default), 
                bbox+mask (2 files, 'mask'), or as padded polygons ('polygon')
            build_items (bool): if True (default), extract and store images for the task
                from the pages; otherwise, just extract the original data from the archive.
            from_tsv_file (str): if set, the data are to be loaded from the given file
                (containing folder is assumed to be the work folder, superceding the
                work_folder option).
            from_page_xml_dir (str): if set, the samples have to be extracted from the 
                raw page data contained in the given directory.
            from_work_folder (str): if set, the samples are to be loaded from the 
                given directory, without prior processing.
            count (int): Stops after extracting {count} image items (for testing 
                purpose only).
            resume_task (bool): If True, the work folder is not purged. Only those page
                items (lines, regions) that not already in the work folder are extracted.
                (Partially implemented: works only for lines.)

        """

        # A dataset resource dictionary needed, unless we build from existing files
        if self.dataset_resource is None and not (from_page_xml_dir or from_tsv_file or from_work_folder):
            raise FileNotFoundError("In order to create a dataset instance, you need either:" +
                                    "\n\t + a valid resource dictionary (cf. 'dataset_resource' class attribute)" +
                                    "\n\t + one of the following options: -from_page_xml_dir, -from_work_folder, -from_tsv_file")
        
        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        super().__init__(root, transform=trf, target_transform=target_transform ) # if target_transform else self.filter_transcription)

        self.root = Path(root) if root else Path(__file__).parents[1].joinpath('data', self.root_folder_basename)

        logger.debug("Root folder: {}".format( self.root ))
        if not self.root.exists():
            self.root.mkdir( parents=True )
            logger.debug("Create root path: {}".format(self.root))

        self.raw_data_folder_path = None
        self.work_folder_path = None # task-dependent

        self.from_line_tsv_file = ''
        if from_tsv_file == '':
            # Local file system with data samples, no archive
            if from_work_folder != '':
                work_folder = from_work_folder
                logger.debug("work_folder="+ work_folder)
                if not Path(work_folder).exists():
                    raise FileNotFoundError(f"Work folder {self.work_folder_path} does not exist. Abort.")
                
            # Local file system with raw page data, no archive 
            elif from_page_xml_dir != '':
                self.raw_data_folder_path = Path( from_page_xml_dir )
                if not self.raw_data_folder_path.exists():
                    raise FileNotFoundError(f"Directory {self.raw_data_folder_path} does not exist. Abort.")
                self.pagexmls = sorted( self.raw_data_folder_path.glob('*.xml'))

            # Online archive
            elif self.dataset_resource is not None:
                # tarball creates its own base folder
                self.raw_data_folder_path = self.root.joinpath( self.dataset_resource['tarball_root_name'] )
                self.download_and_extract( self.root, self.root, self.dataset_resource, extract_pages )
                # input PageXML files are at the root of the resulting tree
                #        (sorting is necessary for deterministic output)
                self.pagexmls = sorted( self.raw_data_folder_path.glob('*.xml'))
            else:
                raise FileNotFoundError("Could not find a dataset source!")
        else:
            # used only by __str__ method
            self.from_tsv_file = from_tsv_file

        # bbox or polygons
        self.shape = shape

        self.data = []
        self.resume_task = resume_task
        build_ok = False if (from_tsv_file!='' or from_work_folder!='' ) else build_items
            
        self._build_task(build_items=build_ok, from_tsv_file=from_tsv_file, 
                         subset=subset, subset_ratios=subset_ratios, 
                         work_folder=work_folder, count=count, 
                         line_padding_style=line_padding_style,
                         expansion_masks=expansion_masks)

        if self.data and not from_tsv_file:
            # Generate a TSV file with one entry per img/transcription pair
            self.dump_data_to_tsv(self.data, Path(self.work_folder_path.joinpath(f"charters_ds_{subset}.tsv")) )
            self._generate_readme("README.md", 
                    { 'subset': subset,
                      'subset_ratios': subset_ratios, 
                      'build_items': build_items, 
                      'count': count, 
                      'from_tsv_file': from_tsv_file,
                      'from_page_xml_dir': from_page_xml_dir,
                      'from_work_folder': from_work_folder,
                      'work_folder': work_folder, 
                      'line_padding_style': line_padding_style,
                      'shape': shape,
                     } )

    def download_and_extract(
            self,
            root: Path,
            raw_data_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        Args:
            root (Path): where to save the archive raw_data_folder_path (Path): where to extract the archive.
            fl_meta (dict): a dictionary with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
            extract (bool): If False (default), skip archive extraction step.

        Returns:
            None

        Raises:
            OSError: the base folder does not exist.
        """
        output_file_path = None
        print(fl_meta)
        # downloadable archive
        if 'url' in fl_meta:
            output_file_path = root.joinpath( fl_meta['tarball_filename'])

            if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
                logger.info("Downloading archive...")
                du.resumable_download(fl_meta['url'], root, fl_meta['tarball_filename'], google=(fl_meta['origin']=='google'))
            else:
                logger.info("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_resource['md5']))
        elif 'file' in fl_meta:
            output_file_path = Path(fl_meta['file'])

        if not raw_data_folder_path.exists() or not raw_data_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract: # and du.check_extracted( raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ) , fl_meta['full-md5'] ):
            logger.info('Found valid file tree in {}: skipping the extraction stage.'.format(str(raw_data_folder_path.joinpath( self.dataset_resource['tarball_root_name'] ))))
            return
        if output_file_path.suffix == '.tgz' or output_file_path.suffixes == [ '.tar', '.gz' ] :
            with tarfile.open(output_file_path, 'r:gz') as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )
        # task description
        elif output_file_path.suffix == '.zip':
            with zipfile.ZipFile(output_file_path, 'r' ) as archive:
                logger.info('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
                archive.extractall( raw_data_folder_path )


    def _build_task( self, 
                   build_items: bool=True, 
                   from_tsv_file: str='',
                   subset: str='train', 
                   subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                   count: int=0, 
                   work_folder: str='', 
                   crop=False,
                   )->None:
        """From the read-only, uncompressed archive files, build the image/GT files required for the task at hand:

        + only creates the files needed for a particular task (train, validate, or test): if more than one subset
          is needed, just initialize a new dataset with desired parameters (work directory, subset)
        + by default, 'train' subset contains 70% of the samples, 'validate', 10%, and 'test', 20%.
        + set samples are randomly picked, but two subsets are guaranteed to be complementary.

        TODO: implement 'from_tsv_file' option.

        Args:
            build_items (bool): if True (default), extract and store
                images for the task from the pages;
            from_tsv_file (str): TSV file from which the data are
                to be loaded (containing folder is assumed to be the work folder, superceding
                the work_folder option). (Default value = '')
            subset (str): 'train', 'validate' or 'test'. (Default value
                = 'train')
            subset_ratios (Tuple[float,float,float]): ratios for
                respective ('train', 'validate', ...) subsets (Default
                value = (.7, 0.1, 0.2))
            count (int): Stops after extracting {count} image items (for
                testing purpose only). (Default value = 0)
            work_folder (str): Where line images and ground truth transcriptions fitting a particular task
                are to be created; default: './MonasteriumHandwritingDatasetSegment'.
            crop (bool): (for segmentation set only) crop text regions from both image and 
                PageXML file. (Default value = False)

        Returns:
            None

        Raises:
            FileNotFoundError: the TSV file passed to the `from_tsv_file` option does not exist.
        """
        self.work_folder_path = Path('.', self.work_folder_name+'Segment') if work_folder=='' else Path( work_folder )
        if not self.work_folder_path.is_dir():
            self.work_folder_path.mkdir(parents=True) 

        if from_tsv_file != '':
            tsv_path = Path( from_tsv_file )
            if tsv_path.exists():
                self.work_folder_path = tsv_path.parent
                # paths are assumed to be absolute
                self.data = self.load_from_tsv( tsv_path, expansion_masks )
                logger.debug("data={}".format( self.data[:6]))
                logger.debug("height: {} type={}".format( self.data[0]['height'], type(self.data[0]['height'])))
                #logger.debug(self.data[0]['polygon_mask'])
            else:
                raise FileNotFoundError(f'File {tsv_path} does not exist!')

        else:
            if work_folder=='':
                self.work_folder_path = Path(self.root, self.work_folder_name+'Segment') 
                logger.debug("Setting default location for work folder: {}".format( self.work_folder_path ))
            else:
                # if work folder is an absolute path, it overrides the root
                self.work_folder_path = self.root.joinpath( work_folder )
                logger.debug("Work folder: {}".format( self.work_folder_path ))

            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir(parents=True)
                logger.debug("Creating work folder = {}".format( self.work_folder_path ))


        if build_items:
            if crop:
                self.data = self._extract_text_regions( self.raw_data_folder_path, self.work_folder_path, count=count )
            else:
                self.data = self._build_page_img_pairs( self.raw_data_folder_path, self.work_folder_path, count=count )
        else:
            logger.info("Building samples from existing images and transcription files in {}".format(self.work_folder_path))
            samples = self.load_items_from_dir( self.work_folder_path )

    @staticmethod
    def load_items_from_dir( work_folder_path: Union[Path,str] ) -> List[dict]:
        """ TODO """
        logger.info("NOT IMPLEMENTED.")
        return []


    @staticmethod
    def load_items_from_tsv( file_path: Union[Path,str] ) -> List[dict]:
        """ TODO """
        logger.info("NOT IMPLEMENTED.")
        return []




    def _generate_readme( self, filename: str, params: dict )->None:
        """Create a metadata file in the work directory.

        Args:
            filename (str): a filepath.
            params (dict): dictionary of parameters passed to the dataset task builder.

        Returns:
            None
        """
        filepath = Path(self.work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n\n\t+ ' + 
                  '\n\t+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)
            print( repr(self), file=of)



    def _build_page_img_pairs(self, raw_data_folder_path:Path,
                                work_folder_path: Path, 
                                text_only:bool=False, 
                                count:int=0, 
                                metadata_format:str='xml') -> List[Tuple[str, str]]:
        """Create a new dataset for segmentation that associate each page image with its metadata.

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            text_only (bool): If True, only generate the transcription files; default is False.
            count (int): Stops after extracting {count} images (for testing purpose). (Default value = 0)
            metadata_format (str): 'xml' (default) or 'json'

        )

        Returns:
            List[Tuple[str,str]]: a list of pairs `(<absolute img filepath>, <absolute transcription filepath>)`
        """
        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there

        if not self.resume_task:
            self._purge( work_folder_path ) 

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


    def _extract_text_regions(self, 
                              raw_data_folder_path: Path, 
                              work_folder_path: Path,
                              text_only=False, 
                              count=0, 
                              metadata_format:str='xml') -> List[Tuple[str, str]]:
        """Crop text regions from original files, and create a new dataset for segmentation where the text

        Args:
            raw_data_folder_path (Path): root of the (read-only) expanded archive.
            work_folder_path (Path): Line images are extracted in this subfolder (relative to the caller's pwd).
            text_only (bool): if True, only generate the GT text files.  Default: False.
            count (int): Stops after extracting {count} images (for testing purpose). (Default value = 0)
            metadata_format (str): `'xml'` (default) or `'json'`

        Returns:
            List[Tuple[str,str]]: a list of pairs `(img_file_path, transcription)`
        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there

        if not self.resume_task:
            self._purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

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
                    #textregion['bbox'] = ImagePath.Path( coordinates ).getbbox()

                    # fix bbox for given region, according to the line points it contains
                    textregion['bbox'] = self._compute_bbox( page, textregion['id'] )
                    if textregion['bbox'] == (0,0,0,0):
                        continue
                    img_path_prefix = work_folder_path.joinpath( f"{xml_id}-{textregion['id']}" )
                    textregion['img_path'] = img_path_prefix.with_suffix('.png')
                    logger.debug('textregion["img_path"] ={} type={}'.format( textregion['img_path'], type(textregion['img_path'])))
                    textregion['size'] = [ textregion['bbox'][i+2]-textregion['bbox'][i]+1 for i in (0,1) ]

                    if not text_only:
                        bbox_img = page_image.crop( textregion['bbox'] )
                        bbox_img.save( textregion['img_path'] )

                    # create a new descriptor file whose a single text region that covers the whole image, 
                    # where line coordinates have been shifted accordingly
                    self._write_region_to_xml( page, ns, textregion )

                    cnt += 1

        return items


    def _compute_bbox(self, page: str, region_id: str ) -> Tuple[int, int, int, int]:
        """In the raw Monasterium/Teklia PageXMl file, baseline and/or textline polygon points
        may be outside the nominal boundaries of their text region. This method computes a
        new bounding box for the given text region, based on the baseline points its contains.

        Args:
            page (str): path to the PageXML file.
            region_id (str): id attribute of the region element in the PageXML file

        Returns:
            Tuple[int,int,int,int]: region's new coordinates, as $(x1, y1, x2, y2)$
        """

        def within(pt, bbox):
            """
            Args:
                pt
                bbox
            """
            return pt[0] >= bbox[0] and pt[0] <= bbox[2] and pt[1] >= bbox[1] and pt[1] <= bbox[3]

        with open(page, 'r') as page_file:

            page_tree = ET.parse( page_file )
            ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
            page_root = page_tree.getroot()

            region_elt = page_root.find( ".//pc:TextRegion[@id='{}']".format( region_id), ns )
            polygon_string=region_elt.find('pc:Coords', ns).get('points')
            coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
            original_bbox = ImagePath.Path( coordinates ).getbbox()

            all_points = []
            valid_lines = 0
            for line_elt in region_elt.findall('pc:TextLine', ns):
                for elt_name in ['pc:Baseline', 'pc:Coords']:
                    elt = line_elt.find( elt_name, ns )
                    if elt_name == 'pc:Baseline' and elt is None:
                        logger.warning('Page {}, region {}: could not find element {} for line {}'.format(
                            page, region_elt.get('id'), elt_name, line_elt.get('id')))
                        continue
                    valid_lines += 1
                    #logger.debug( elt.get('points').split(','))
                    all_points.extend( [ tuple(map(int, pt.split(','))) for pt in elt.get('points').split(' ') ])
            if not valid_lines:
                return (0,0,0,0)
            not_ok = [ p for p in all_points if not within( p, original_bbox) ]
            if not_ok:
                logger.warning("File {}: invalid points for textregion {}: {} -> extending bbox accordingly".format(page, region_id, not_ok))
            bbox = ImagePath.Path( all_points ).getbbox()
            logger.debug("region {}, bbox={}".format( region_id, bbox))
            return bbox


    def _write_region_to_xml( self, page: str, ns: str, textregion: dict )->None:
        """From the given text region data, generates a new PageXML file.

        TODO: fix bug in ImageFilename attribute E.g. NA-RM_14240728_2469_r-r1..jpg

        Args:
            page (str): path of the pageXML file to generate.
            ns (str): namespace.
            textregion (dict): a dictionary of text region attributes.

        Returns:
            None
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


    @staticmethod
    def _split_set(samples: object, ratios: Tuple[float, float, float], subset: str) -> List[object]:
        """Split a dataset into 3 sets: train, validation, test.

        Args:
            samples (object): any dataset sample.
            ratios (Tuple[float, float, float]): respective proportions for possible subsets
            subset (str): subset to be build  ('train', 'validate', or 'test')

        Returns:
            List[object]: a list of samples.

        Raises:
            ValueError: The subset type does not exist.
        """

        random.seed(10)
        logger.debug("Splitting set of {} samples with ratios {}".format( len(samples), ratios))

        if 1.0 in ratios:
            return list( samples )
        if subset not in ('train', 'validate', 'test'):
            raise ValueError("Incorrect subset type: choose among 'train', 'validate', and 'test'.")

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


    def __getitem__(self, index) -> Dict[str, Union[Tensor, int, str]]:
        """Callback function for the iterator.
        TODO: complete for segmentation tasks.

        Args:
            index (int): item index.

        Returns:
            dict[str,Union[Tensor,int,str]]: a sample dictionary
        """
        img_path = self.data[index]['img']
        
        assert isinstance(img_path, Path) or isinstance(img_path, str)

        # In the sample, image filename replaced with 
        # - file id ('id')
        # - tensor ('img')
        # - dimensions of transformed image ('height' and 'width')
        # 
        sample = self.data[index].copy()
        
        sample['img'] = Image.open( img_path, 'r')

        sample['id'] = Path(img_path).name
        logger.debug('sample='.format(index), sample)
        return sample

    def __getitems__(self, indexes: list ) -> List[dict]:
        """To help with batching.

        Args:
            indexes (list): a list of indexes.

        Returns:
            List[dict]: a list of samples.
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns:
            int: number of data points.
        """
        return len( self.data )


    def _purge(self, folder: str) -> int:
        """Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the TSV file.

        Args:
            folder (str): Name of the subfolder to _purge (relative the caller's pwd

        Returns:
            int: number of deleted files.
        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
            item.unlink()
            cnt += 1
        return cnt

    def __repr__(self) -> str:

        summary = '\n'.join([
                    f"Root folder:\t{self.root}",
                    f"Files extracted in:\t{self.raw_data_folder_path}",
                    f"Line shape: {self.shape}",
                    f"Work folder:\t{self.work_folder_path}",
                    f"Data points:\t{len(self.data)}",
                    "Stats:",
                    f"{self.dataset_stats(self.data)}" if self.data else 'No data',])

        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")


    def count_items(self, folder) -> Tuple[int, int]:
        """Count dataset items in the given folder.

        Args:
            folder (str): a directory path.

        Returns:
            Tuple[int,int]: a pair `(<number of GT files>, <number of image files>)`
        """
        return (
                len( [ i for i in Path(folder).glob('*.xml') ] ),
                len( [ i for i in Path(folder).glob('*.img.*') ] )
                )



class MonasteriumDataset(ChartersDataset):
    """A subset of Monasterium charter images and their meta-data (PageXML).

        + its core is a set of charters segmented and transcribed by various contributors, mostly by correcting Transkribus-generated data.
        + it has vocation to grow through in-house, DiDip-produced transcriptions.
    """

    dataset_resource = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'tarball_filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
            'tarball_root_name': 'MonasteriumTekliaGTDataset',
            'comment': 'A clean, terse dataset, with no use of Unicode abbreviation marks.',
    }

    work_folder_name="MonasteriumHandwritingDataset"

    root_folder_basename="Monasterium"

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)


class KoenigsfeldenDataset(ChartersDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662',
            'comment': 'Transcriptions have been cleaned up (removal of obvious junk or non-printable characters, as well a redundant punctuation marks---star-shaped unicode symbols); unicode-abbreviation marks have been expanded.',
    }

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="Koenigsfelden"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription




class KoenigsfeldenDatasetAbbrev(ChartersDataset):
    """A subset of charters from the Koenigsfelden abbey, covering a wide range of handwriting style.
        The data have been compiled from raw Transkribus exports.
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/koenigsfelden_abbey_1308-1662/koenigsfelden_1308-1662.tar.gz",
            'tarball_filename': 'koenigsfelden_1308-1662_abbrev.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Koenigsfelden ground truth data',
            'origin': 'local',
            'tarball_root_name': 'koenigsfelden_1308-1662_abbrev',
            'comment': 'Similar to the KoenigsfeldenDataset, with a notable difference: Unicode abbreviations have been kept.',
    }

    work_folder_name="KoenigsfeldenHandwritingDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="KoenigsfeldenAbbrev"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)

        #self.target_transform = self.filter_transcription


class NurembergLetterbooks(ChartersDataset):
    """
    Nuremberg letterbooks (15th century).
    """

    dataset_resource = {
            'file': f"{os.getenv('HOME')}/tmp/data/nuremberg_letterbooks/nuremberg_letterbooks.tar.gz",
            'tarball_filename': 'nuremberg_letterbooks.tar.gz',
            'md5': '9326bc99f9035fb697e1b3f552748640',
            'desc': 'Nuremberg letterbooks ground truth data',
            'origin': 'local',
            'tarball_root_name': 'nuremberg_letterbooks',
            'comment': 'Numerous struck-through lines (masked)'
    }

    work_folder_name="NurembergLetterbooksDataset"
    "This prefix will be used when creating a work folder."

    root_folder_basename="NurembergLetterbooks"
    "This is the root of the archive tree."

    def __init__(self, *args, **kwargs ):

        super().__init__( *args, **kwargs)




def dummy():
    """"""
    return True
