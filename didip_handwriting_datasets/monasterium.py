import sys
import warnings
import random
import tarfile
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
import skimage as ski

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
    """ """
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
"""

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)

# this is the tarball's top folder, automatically created during the extraction  (not configurable)
tarball_root_name="MonasteriumTekliaGTDataset"    
work_folder_name="MonasteriumHandwritingDataset"
root_folder_basename="Monasterium"
alphabet_tsv_name="alphabet.tsv"




class MonasteriumDataset(VisionDataset):
    """ Dataset for charters.

        :param dataset_file: 
            meta-data (URL, archive name, type of repository).
        :type dataset_file: dict

        :param default_alphabet:
            a many-to-one alphabet in list form, to be used if no other alphabet is passed to the initialization function.
        :type default_alphabet: list
    """

    dataset_file = {
            #'url': r'https://cloud.uni-graz.at/apps/files/?dir=/DiDip%20\(2\)/CV/datasets&fileid=147916877',
            'url': r'https://drive.google.com/uc?id=1hEyAMfDEtG0Gu7NMT7Yltk_BAxKy_Q4_',
            'filename': 'MonasteriumTekliaGTDataset.tar.gz',
            'md5': '7d3974eb45b2279f340cc9b18a53b47a',
            'full-md5': 'e720bac1040523380921a576f4cc89dc',
            'desc': 'Monasterium ground truth data (Teklia)',
            'origin': 'google',
    }

    default_alphabet = [' ',
                       [',', '.', ':', ';'],
                       ['-', '¬', '—'],
                       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       ['A', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ă', 'Ą'],
                       'B',
                       ['C', 'Ç', 'Ć', 'Ĉ', 'Ċ', 'Č'],
                       ['D', 'Ð', 'Ď', 'Đ'],
                       ['E', 'È', 'É', 'Ê', 'Ë', 'Ē', 'Ĕ', 'Ė', 'Ę', 'Ě'],
                       'F',
                       ['G', 'Ĝ', 'Ğ', 'Ġ', 'Ģ'],
                       ['H', 'Ĥ', 'Ħ'],
                       ['I', 'J', 'Ì', 'Í', 'Î', 'Ï', 'Ĩ', 'Ī', 'Ĭ', 'Į', 'İ', 'Ĳ', 'Ĵ'],
                       ['K', 'Ķ'],
                       ['L', 'Ĺ', 'Ļ', 'Ľ', 'Ŀ', 'Ł'],
                       'M',
                       ['N', 'Ñ', 'Ń', 'Ņ', 'Ň', 'Ŋ'],
                       ['O', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ō', 'Ŏ', 'Ő', 'Œ'],
                       'P',
                       'Q',
                       ['R', 'Ŕ', 'Ŗ', 'Ř'],
                       ['S', 'ß', 'Ś', 'Ŝ', 'Ş', 'Š'],
                       ['T', 'Ţ', 'Ť', 'Ŧ'],
                       ['U', 'V', 'Ù', 'Ú', 'Û', 'Ü', 'Ũ', 'Ū', 'Ŭ', 'Ů', 'Ű', 'Ų'],
                       ['W', 'Ŵ'],
                       'X',
                       ['Y', 'Ŷ', 'Ÿ'],
                       ['Z', 'Ź', 'Ż', 'Ž'],
                       ['a', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ā', 'ă', 'ą'],
                       'b',
                       ['c', 'ç', 'ć', 'ĉ', 'ċ', 'č'],
                       ['d', 'ð', 'ď', 'đ'],
                       ['e', 'è', 'é', 'ê', 'ë', 'ē', 'ĕ', 'ė', 'ę', 'ě'],
                       'f',
                       ['g', 'ĝ', 'ğ', 'ġ', 'ģ'],
                       ['h', 'ĥ', 'ħ'],
                       ['i', 'j', 'ì', 'í', 'î', 'ï', 'ĩ', 'ī', 'ĭ', 'į', 'ı', 'ĳ', 'ĵ'],
                       ['k', 'ķ', 'ĸ'],
                       ['l', 'ĺ', 'ļ', 'ľ', 'ŀ', 'ł'],
                       'm',
                       ['n', 'ñ', 'ń', 'ņ', 'ň', 'ŉ', 'ŋ'],
                       ['o', 'ò', 'ó', 'ô', 'õ', 'ö', 'ō', 'ŏ', 'ő', 'œ'],
                       'p',
                       'q',
                       ['r', 'ŕ', 'ŗ', 'ř'],
                       ['s', 'ś', 'ŝ', 'ş', 'š'],
                       ['t', 'ţ', 'ť', 'ŧ'],
                       ['u', 'v', 'ù', 'ú', 'û', 'ü', 'ũ', 'ū', 'ŭ', 'ů', 'ű', 'ų'],
                       ['w', 'ŵ'],
                       'x',
                       ['y', 'ý', 'ÿ', 'ŷ'],
                       ['z', 'ź', 'ż', 'ž']]


    def __init__( self,
                root: str='',
                work_folder: str = '', # here further files are created, for any particular task
                subset: str = 'train',
                subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract_pages: bool = False,
                from_tsv_file: str = '',
                build_items: bool = True,
                task: str = '',
                shape: str = '',
                count: int = 0,
                alphabet_tsv: str = None,
                ) -> None:
        """Initialize a dataset instance.

        :param root: 
            Where the archive is to be downloaded and the subfolder containing original files (pageXML documents and page images)
            is to be created. Default: subfolder `data/Monasterium' in this project's directory.
        :type root: str
        :param work_folder:
            Where line images and ground truth transcriptions fitting a particular task are to be created;
            default: '<root>/MonasteriumHandwritingDatasetHTR'; if parameter is a relative path, the work folder is created under <root>;
            an absolute path overrides this. For HTR task, the work folder also contains the alphabet in TSV form.
        :type work_folder: str

        :param subset: 
            'train' (default), 'validate' or 'test'.
        :type subset: str

        :param subset_ratios:
            ratios for respective ('train', 'validate', ...) subsets
        :type subset_ratios: Tuple[float, float, float]

        :param transform: 
            Function to apply to the PIL image at loading time.
        :type transform: Callable

        :param target_transform: 
            Function to apply to the transcription ground truth at loading time.
        :type target_transform: Callable

        :param extract_pages: 
            if True, extract the archive's content into the base folder no matter what; 
            otherwise (default), check first for a file tree with matching name and checksum.
        :type extract_pages: bool

        :param task: 
            'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation = cropped TextRegion images,
            with corresponding PageXML files.  If '' (default), the dataset archive is extracted but no actual data get built.
        :type task: str

        :param shape: 
            'bbox' (default) for line bounding boxes or 'polygons'
        :type shape: str

        :param build_items: 
            if True (default), extract and store images for the task from the pages; otherwise,
            just extract the original data from the archive.
        :type build_items: bool

        :param from_tsv_file: 
            TSV file from which the data are to be loaded (containing folder is assumed to be the work folder,
            superceding the work_folder option).
        :type from_tsv_file: str

        :param count: 
            Stops after extracting {count} image items (for testing purpose only).
        :type count: int

        :param alphabet_tsv: 
            TSV file containing the alphabet
        :type alphabet_tsv: str
        """
        
        trf = v2.PILToTensor()
        if transform:
            trf = v2.Compose( [ v2.PILToTensor(), transform ] )

        super().__init__(root, transform=trf, target_transform=target_transform )

        self.root = Path(root) if root else Path(__file__).parent.joinpath('data', root_folder_basename)
        logger.debug("Root folder: {}".format( self.root ))
        if not self.root.exists():
            self.root.mkdir( parents=True )
            logger.debug("Create root path: {}".format(self.root))

        self.work_folder_path = None # task-dependent
        # tarball creates its own base folder
        self.raw_data_folder_path = self.root.joinpath( tarball_root_name )

        self.from_tsv_file = ''
        if from_tsv_file == '':
            self.download_and_extract( self.root, self.root, self.dataset_file, extract_pages)
        else:
            # used only by __str__ method
            self.from_tsv_file = from_tsv_file

        # input PageXML files are at the root of the resulting tree
        # (sorting is necessary for deterministic output)
        self.pagexmls = sorted( Path(self.root, tarball_root_name ).glob('*.xml'))

        self.shape = shape

        self.data = []

        # Used only for HTR tasks: initialized by _build_task()
        self.alphabet = None

        self._task = ''
        if (task != ''):
            self._task = task # for self-documentation only
            build_ok = build_items if from_tsv_file == '' else False
            self._build_task( task, build_items=build_ok, from_tsv_file=from_tsv_file, 
                             subset=subset, subset_ratios=subset_ratios, 
                             work_folder=work_folder, count=count, alphabet_tsv=alphabet_tsv )

            logger.debug("data={}".format( self.data[:6]))

    def _build_task( self, 
                   task: str='htr',
                   build_items: bool=True, 
                   from_tsv_file: str='',
                   subset: str='train', 
                   subset_ratios: Tuple[float,float,float]=(.7, 0.1, 0.2),
                   count: int=0, 
                   work_folder: str='', 
                   crop=False,
                   alphabet_tsv='',
                   )->None:
        """From the read-only, uncompressed archive files, build the image/GT files required for the task at hand:

        + only creates the files needed for a particular task (train, validate, or test): if more than one subset
          is needed, just initialize a new dataset with desired parameters (work directory, subset)
        + by default, 'train' subset contains 70% of the samples, 'validate', 10%, and 'test', 20%.
        + set samples are randomly picked, but two subsets are guaranteed to be complementary.

        :param task: 
            'htr' for HTR set = pairs (line, transcription), 'segment' for segmentation (Default value = 'htr')
        :type task: str
        :param build_items: 
            if True (default), extract and store images for the task from the pages;
        :type build_items: bool
        :param from_tsv_file: 
            TSV file from which the data are to be loaded (containing folder is
                                 assumed to be the work folder, superceding the work_folder option). (Default value = '')
        :type from_tsv_file: str
        :param subset: 
            'train', 'validate' or 'test'. (Default value = 'train')
        :type subset: str
        :param subset_ratios: 
            ratios for respective ('train', 'validate', ...) subsets (Default value = (.7, 0.1, 0.2))
        :type subset_ratios: Tuple[float,float,float]
        :param count: 
            Stops after extracting {count} image items (for testing purpose only). (Default value = 0)
        :type count: int
        :param work_folder: 
            Where line images and ground truth transcriptions fitting a particular task
                               are to be created; default: './MonasteriumHandwritingDatasetHTR'.
        :type work_folder: str
        :param crop: 
            (for segmentation set only) crop text regions from both image and PageXML file. (Default value = False)
        :type crop: bool
        :param alphabet_tsv: 
            TSV file containing the alphabet (Default value = '')
        :type alphabet_tsv: str

        :rtype: None

        :raises FileNotFoundError: the TSV file passed to the `from_tsv_file` option does not exist, or
               the specified TSV alphabet does not exist.

        """
        if task == 'htr':
            
            if crop:
                self.logger.warning("Warning: the 'crop' [to WritingArea] option ignored for HTR dataset.")
            
            # create from existing TSV files - passed directory that contains:
            # + image to GT mapping (TSV)
            # + alphabet.tsv
            if from_tsv_file != '':
                tsv_path = Path( from_tsv_file )
                if tsv_path.exists():
                    self.work_folder_path = tsv_path.parent
                    # paths are assumed to be absolute
                    self.data = self.load_from_tsv( tsv_path )
                    logger.debug("_build_task(): data={}".format( self.data[:6]))
                    logger.debug("_build_task(): height: {} type={}".format( self.data[0]['height'], type(self.data[0]['height'])))
                    #logger.debug(self.data[0]['polygon_mask'])
                else:
                    raise FileNotFoundError(f'File {tsv_path} does not exist!')

            else:
                if work_folder=='':
                    self.work_folder_path = Path(self.root, work_folder_name+'HTR') 
                    logger.debug("Setting default location for work folder: {}".format( self.work_folder_path ))
                else:
                    # if work folder is an absolute path, it overrides the root
                    self.work_folder_path = self.root.joinpath( work_folder )
                    logger.debug("Work folder: {}".format( self.work_folder_path ))

                if not self.work_folder_path.is_dir():
                    self.work_folder_path.mkdir(parents=True)
                    logger.debug("Creating work folder = {}".format( self.work_folder_path ))

                # samples: all of them! (Splitting into subset happens in a ulterior step.)
                if build_items:
                    samples = self._extract_lines( self.raw_data_folder_path, self.work_folder_path, count=count, shape=self.shape )
                else:
                    logger.info("Building samples from existing images and transcription files in {}".format(self.work_folder_path))
                    samples = self.load_line_items_from_dir( self.work_folder_path )

                self.data = self._split_set( samples, ratios=subset_ratios, subset=subset)
                logger.info(f"Subset contains {len(self.data)} samples.")

                
                # Generate a TSV file with one entry per img/transcription pair
                self.dump_data_to_tsv(self.data, Path(self.work_folder_path.joinpath(f"monasterium_ds_{subset}.tsv")) )
                self._generate_readme("README.md", 
                        { 'subset': subset, 'subset_ratios': subset_ratios, 'build_items': build_items, 
                          'task': task, 'crop': crop, 'count': count, 'from_tsv_file': from_tsv_file,
                          'work_folder': work_folder, 'alphabet_tsv': alphabet_tsv } )

                # serialize the alphabet into the work folder
                logger.debug("Serialize default (hard-coded) alphabet into {}".format(self.work_folder_path.joinpath('alphabet.tsv')))
                alphabet.Alphabet( self.default_alphabet ).to_tsv( self.work_folder_path.joinpath('alphabet.tsv'))
                #shutil.copy(self.root.joinpath( alphabet_tsv_name ), self.work_folder_path )
            
            # load alphabet
            alphabet_tsv_input = Path( alphabet_tsv ) if alphabet_tsv else self.work_folder_path.joinpath( alphabet_tsv_name )
            if not alphabet_tsv_input.exists():
                raise FileNotFoundError("Alphabet file: {}".format( alphabet_tsv_input))
            logger.debug('alphabet path: {}'.format( str(alphabet_tsv_input)))
            self.alphabet = alphabet.Alphabet( alphabet_tsv_input )
                

        elif task == 'segment':
            self.work_folder_path = Path('.', work_folder_name+'Segment') if work_folder=='' else Path( work_folder )
            if not self.work_folder_path.is_dir():
                self.work_folder_path.mkdir(parents=True) 

            if build_items:
                if crop:
                    self.data = self._extract_text_regions( self.raw_data_folder_path, self.work_folder_path, count=count )
                else:
                    self.data = self._build_page_lines_pairs( self.raw_data_folder_path, self.work_folder_path, count=count )


    @staticmethod
    def load_line_items_from_dir( work_folder_path: Union[Path,str] ) -> List[dict]:
        """Construct a list of samples from a directory that has been populated with
        line images and line transcriptions

        :param work_folder_path: 
            a folder containing images (`*.png`) and transcription files (`*.gt.txt`)
        :type work_folder_path: Union[Path,str]

        :returns: a list of samples.
        :rtype: List[dict]

        """
        samples = []
        if type(work_folder_path) is str:
            work_folder_path = Path( work_folder_path )
        for img_file in work_folder_path.glob('*.png'):
            sample=dict()
            logger.debug(img_file)            
            gt_file_name = img_file.with_suffix('.gt.txt')
            sample['img']=img_file
            img = Image.open( img_file, 'r')
            sample['width'], sample['height'] = img.size
            
            with open(gt_file_name, 'r') as gt_if:
                sample['transcription']=gt_if.read().rstrip()
            samples.append( sample )

        logger.debug(f"Loaded {len(samples)} samples from {work_folder_path}")
        return samples
                

    @staticmethod
    def dataset_stats( samples: List[dict] ) -> str:
        """Compute basic stats about sample sets.
        
        + avg, median, min, max on image heights and widths
        + avg, median, min, max on transcriptions
        + effective character set + which subset of the default alphabet is being used

        :param samples: 
            a list of samples.
        :type samples: List[dict]

        :returns: a string.
        :rtype: str

        """
        heights = np.array([ s['height'] for s in samples  ], dtype=int)
        widths = np.array([ s['width'] for s in samples  ], dtype=int)
        gt_lengths = np.array([ len(s['transcription']) for s in samples  ], dtype=int)

        height_stats = [ int(s) for s in(np.mean( heights ), np.median(heights), np.min(heights), np.max(heights))]
        width_stats = [int(s) for s in (np.mean( widths ), np.median(widths), np.min(widths), np.max(widths))]
        gt_length_stats = [int(s) for s in (np.mean( gt_lengths ), np.median(gt_lengths), np.min(gt_lengths), np.max(gt_lengths))]

        stat_list = ('Mean', 'Median', 'Min', 'Max')
        row_format = "{:>10}" * (len(stat_list) + 1)
        return '\n'.join([
            row_format.format("", *stat_list),
            row_format.format("Img height", *height_stats),
            row_format.format("Img width", *width_stats),
            row_format.format("GT length", *gt_length_stats),
        ])



    def _generate_readme( self, filename: str, params: dict )->None:
        """Create a metadata file in the work directory.
        
        :param filename: 
            a filepath. 
        :type filename: str
        :param params: 
            dictionary of parameters passed to the dataset task builder.
        :type params: dict

        :rtype: None

        """
        filepath = Path(self.work_folder_path, filename )
        
        with open( filepath, "w") as of:
            print('Task was built with the following options:\n' + 
                  '\n\t+ '.join( [ f"{k}={v}" for (k,v) in params.items() ] ),
                  file=of)


    def download_and_extract(
            self,
            root: Path,
            raw_data_folder_path: Path,
            fl_meta: dict,
            extract=False) -> None:
        """Download the archive and extract it. If a valid archive already exists in the root location,
        extract only.

        :param root: 
            where to save the archive
        :type root: Path
        :param raw_data_folder_path: 
            where to extract the archive.
        :type raw_data_folder_path: Path
        :param fl_meta: 
            a dictionary with file meta-info (keys: url, filename, md5, full-md5, origin, desc)
        :type fl_meta: dict
        :param extract:
            If False (default), skip archive extraction step.
        :type extract: bool

        :rtype: None

        :raises OSError: the base folder does not exist.

        """
        output_file_path = root.joinpath( fl_meta['filename'])

        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            logger.info("Downloading archive...")
            du.resumable_download(fl_meta['url'], root, fl_meta['filename'], google=(fl_meta['origin']=='google'))
        else:
            logger.info("Found valid archive {} (MD5: {})".format( output_file_path, self.dataset_file['md5']))

        if not raw_data_folder_path.exists() or not raw_data_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        # skip if archive already extracted (unless explicit override)
        if not extract and du.check_extracted( raw_data_folder_path.joinpath( tarball_root_name ) , fl_meta['full-md5'] ):
            logger.info('Found valid file tree in {}: skipping the extraction stage.'.format(str(raw_data_folder_path.joinpath( tarball_root_name ))))
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


    def _purge(self, folder: str) -> int:
        """Empty the line image subfolder: all line images and transcriptions are
        deleted, as well as the TSV file.

        :param folder:
            Name of the subfolder to _purge (relative the caller's pwd
        :type folder: str

        :returns: number of deleted files.
        :rtype: int

        """
        cnt = 0
        for item in [ f for f in Path( folder ).iterdir() if not f.is_dir()]:
            item.unlink()
            cnt += 1
        return cnt


    @staticmethod
    def dump_data_to_tsv(samples: List[dict], file_path: str='', all_path_style=False) -> None:
        """Create a CSV file with all tuples (`<line image absolute path>`, `<transcription>`, `<height>`, `<width>` `[<polygon points]`).
        Height and widths are the original heights and widths.

        :param samples: 
            dataset samples.
        :type samples: List[dict]
        :param file_path: 
            A TSV (absolute) file path (Default value = '')
        :type file_path: str
        :param all_path_style: 
            list GT file name instead of GT content. (Default value = False)
        :type all_path_style: bool

        :rtype: None

        """
        if file_path == '':
            for sample in samples:
                # note: TSV only contains the image file name (load_from_tsv() takes care of applying the correct path prefix)
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                logger.debug("{}\t{}\t{}\t{}".format( img_path, 
                      gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'), int(height), int(width)))
            return
        with open( file_path, 'w' ) as of:
            for sample in samples:
                img_path, gt, height, width = sample['img'].name, sample['transcription'], sample['height'], sample['width']
                #logger.debug('{}\t{}'.format( img_path, gt, height, width ))
                of.write( '{}\t{}\t{}\t{}'.format( img_path,
                                             gt if not all_path_style else Path(img_path).with_suffix('.gt.txt'),
                                             int(height), int(width) ))
                if 'polygon_mask' in sample and sample['polygon_mask'] is not None:
                    of.write('\t{}'.format( sample['polygon_mask'] ))
                of.write('\n')
                                            

    @staticmethod
    def load_from_tsv(file_path: Path) -> List[dict]:
        """Load samples (as dictionaries) from an existing TSV file. Each input line may be either a tuple::
        
                <img file path> <transcription text> <height> <width> [<polygon points>]
        
        or::
        
                <img file path> <transcription file path> <height> <width> [<polygon points>]

        :param file_path: 
            A file path (relative to the caller's pwd).
        :type file_path: Path

        :returns: A list of dictionaries of the form::

                {'img': <img file path>,
                 'transcription': <transcription text>,
                 'height': <original height>,
                 'width': <original width>,
                 'mask': <mask for unpadded part of the img>}}
        :rtype: List[dict]

        :raises ValueError: the TSV file has an incorrect number of fields.

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
            #logger.debug('load_from_tsv(): type(img_path)={} type(height)={}'.format( type(img_path), type(height)))
            all_path_style = True if Path(file_or_text).exists() else False
            infile.seek(0) 

            if not all_path_style:

                def tsv_to_dict( tsv_line ):
                    """

                    :param tsv_line: 

                    """
                    img, transcription, height, width, polygon_mask = [ None ] * 5
                    if has_polygon:
                        fields = tsv_line[:-1].split('\t')
                        logger.debug("fields={}".format(fields))
                        if len(fields)<5:
                            raise ValueError("Incorrect number of fields: {}".format(fields))

                        img, transcription, height, width, polygon_mask = tsv_line[:-1].split('\t')
                        #logger.debug('tsv_to_dict(): type(height)=', type(height))

                        s = {'img': str(work_folder_path.joinpath( img )), 'transcription': transcription,
                                'height': int(height), 'width': int(width), 'polygon_mask': 'None' if polygon_mask == '-' else eval(polygon_mask)  }
                        #logger.debug('type(s[img])={} type(s[height]={}'.format( type(s['img']), type(s['height'])))
                        return s
                    else:
                        img, transcription, height, width = tsv_line[:-1].split('\t')
                        s = {'img': str(work_folder_path.joinpath( img )), 'transcription': transcription,
                                'height': int(height), 'width': int(width) }
                        #logger.debug('type(s[img])={} type(s[height]={}'.format( type(s['img']), type(s['height'])))
                        return s

                samples = [ tsv_to_dict(s) for s in infile ]
                #logger.debug("tsv_to_dict(): samples=", samples)
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


    def _build_page_lines_pairs(self, raw_data_folder_path:Path,
                                work_folder_path: Path, 
                                text_only:bool=False, 
                                count:int=0, 
                                metadata_format:str='xml') -> List[Tuple[str, str]]:
        """Create a new dataset for segmentation that associate each page image with its metadata.

        :param raw_data_folder_path: 
            root of the (read-only) expanded archive.
        :type raw_data_folder_path: Path
        :param work_folder_path: 
            Line images are extracted in this subfolder (relative to the caller's pwd).
        :type work_folder_path: Path
        :param text_only: 
            If True, only generate the transcription files; default is False.
        :type text_only: bool
        :param count: 
            Stops after extracting {count} images (for testing purpose). (Default value = 0)
        :type count: int
        :param metadata_format: 
            'xml' (default) or 'json'
        :type metadata_format: str

        :returns: a list of pairs `(<absolute img filepath>, <absolute transcription filepath>)`
        :rtype: List[Tuple[str,str]]

        """
        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there
        self._purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

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
        region image has a corresponding, new PageXML decriptor.

        :param raw_data_folder_path:
            root of the (read-only) expanded archive.
        :type raw_data_folder_path: Path
        :param work_folder_path:
            Line images are extracted in this subfolder (relative to the caller's pwd).
        :type work_folder_path: Path
        :param text_only: 
            if True, only generate the GT text files. Default: False.
        :type text_only: bool
        :param count:
            Stops after extracting {count} images (for testing purpose). (Default value = 0)
        :type count: int
        :param metadata_format: 
            `'xml'` (default) or `'json'`
        :type metadata_format: str

        :returns: a list of pairs `(img_file_path, transcription)`
        :rtype: List[Tuple[str,str]]

        """
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there
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
                    #textregion['bbox'] = IP.Path( coordinates ).getbbox()

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

        :param page:
            path to the PageXML file.
        :type page: str
        :param region_id: 
            id attribute of the region element in the PageXML file
        :type region_id: str

        :returns: region's new coordinates, as (x1, y1, x2, y2)
        :rtype: Tuple[int,int,int,int]

        """

        def within(pt, bbox):
            """

            :param pt: 
            :param bbox: 

            """
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
            bbox = IP.Path( all_points ).getbbox()
            logger.debug("region {}, bbox={}".format( region_id, bbox))
            return bbox


    def _write_region_to_xml( self, page: str, ns: str, textregion: dict )->None:
        """From the given text region data, generates a new PageXML file.
        
        TODO: fix bug in ImageFilename attribute E.g. NA-RM_14240728_2469_r-r1..jpg

        :param page: 
            path of the pageXML file to generate.
        :type page: str
        :param ns: 
            namespace.
        :type ns: str
        :param textregion: 
            a dictionary of text region attributes.
        :type textregion: dict

        :rtype: None

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


    def _extract_lines(self, raw_data_folder_path: Path,
                        work_folder_path: Path, 
                        shape: str='bbox',
                        text_only=False,
                        count=0) -> List[Dict[str, Union[Tensor,str,int]]]:
        """Generate line images from the PageXML files and save them in a local subdirectory
        of the consumer's program.

        :param raw_data_folder_path: 
            root of the (read-only) expanded archive.
        :type raw_data_folder_path: Path
        :param work_folder_path: 
            Line images are extracted in this subfolder (relative to the caller's pwd).
        :type work_folder_path: Path
        :param shape: 
            Extract lines as bboxes (default) or as polygon-within-bbox.
        :type shape: str
        :param text_only: 
            Store only the transcriptions (*.gt.txt files). (Default value = False)
        :type text_only: bool
        :param count: 
            Stops after extracting {count} images (for testing purpose). (Default value = 0)
        :type count: int

        :returns: An array of dictionaries of the form::
                {'img': <absolute img_file_path>,
                 'transcription': <transcription text>,
                 'height': <original height>,
                 'width': <original width>}
        :rtype: List[Dict[str,Union[Tensor,str,int]]]

        """
        logger.debug("_extract_lines()")
        # filtering out Godzilla-sized images (a couple of them)
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        Path( work_folder_path ).mkdir(exist_ok=True, parents=True) # always create the subfolder if not already there
        self._purge( work_folder_path ) # ensure there are no pre-existing line items in the target directory

        gt_lengths = []
        img_sizes = []
        cnt = 0 # for testing purpose

        samples = [] # each sample is a dictionary {'img': <img file path> , 'transcription': str,
                     #                              'height': int, 'width': int}

        for page in tqdm(self.pagexmls):

            xml_id = Path( page ).stem
            img_path = Path( raw_data_folder_path, f'{xml_id}.jpg')
            #logger.debug( img_path )

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

                    #logger.debug("_extract_lines():", samples[-1])
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

                            # 1. Open image as a  np.array
                            img_hwc = np.array( bbox_img )
                            # 2. Compute 2D Boolean polygon mask with ski.draw, from points
                            leftx, topy = textline['bbox'][:2]
                            transposed_coordinates = np.array([ (x-leftx, y-topy) for x,y in coordinates ])[:,::-1]
                            boolean_mask = ski.draw.polygon2mask( img_hwc.shape[:2], transposed_coordinates )
                            # 3. For each of the 3 channels:
                            for ch in (0,1,2):
                                # - compute median from mask-in
                                med = np.median( img_hwc[:,:,ch][boolean_mask] )
                                # - color mask-out with median
                                median_background = np.full( boolean_mask.shape, med ) * np.logical_not( boolean_mask )
                                # - mask-in + mask-out
                                img_hwc[:,:,ch]=img_hwc[:,:,ch]*boolean_mask  + median_background

                            textline['polygon']=transposed_coordinates

                            Image.fromarray(img_hwc).save( Path( textline['img_path'] ))


                    sample = {'img': textline['img_path'], 'transcription': textline['transcription'], \
                               'height': textline['height'], 'width': textline['width'] }
                    #logger.debug("_extract_lines(): sample=", sample)
                    if textline['polygon'] is not None:
                        sample['polygon_mask'] = textline['polygon']
                    samples.append( sample )

                    with open( img_path_prefix.with_suffix('.gt.txt'), 'w') as gt_file:
                        gt_file.write( textline['transcription'] )
                        gt_lengths.append(len( textline['transcription']))

                    cnt += 1

        return samples
    


    @staticmethod
    def _split_set(samples: object, ratios: Tuple[float, float, float], subset: str) -> List[object]:
        """Split a dataset into 3 sets: train, validation, test.

        :param samples: 
            any dataset sample.
        :type samples: object
        :param ratios: 
            respective proportions for possible subsets
        :type ratios: Tuple[float, float, float]
        :param subset: 
            subset to be build  ('train', 'validate', or 'test')
        :type subset: str

        :returns: a list of samples.
        :rtype: List[object]

        :raises ValueError: The subset type does not exist.
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

        :param index: 
            item index.
        :type index: int

        :returns: a sample dictionary
        :rtype: dict[str,Union[Tensor,int,str]]
        """
        img_path = self.data[index]['img']
        
        #polygon_mask = self.data[index]['polygon_mask'] if 'polygon_mask' in self.data[index] else None

        assert isinstance(img_path, Path) or isinstance(img_path, str)

        # In the sample, image filename replaced with 
        # - file id ('id')
        # - tensor ('img')
        # - dimensions of transformed image ('height' and 'width')
        # 
        sample = self.data[index].copy()
        sample['img'] = Image.open( img_path, 'r')
        # if resized, should store new height and width
        sample = self.transform( sample )
        sample['id'] = Path(img_path).name
        logger.debug('sample='.format(index), sample)
        return sample

    def __getitems__(self, indexes: list ) -> List[dict]:
        """ To help with batching.

        :param indexes: 
            a list of indexes.
        :type indexes: list

        :returns: a list of samples.
        :rtype: List[dict]
        """
        return [ self.__getitem__( idx ) for idx in indexes ]


    def __len__(self) -> int:
        """
        Number of samples in the dataset.

        :returns: number of data points.
        :rtype: int
        """
        return len( self.data )

    @property
    def task( self ):
        """ """
        if self._task == 'htr':
            return "HTR"
        if self._task == 'segment':
            return "Segmentation"
        return "None defined."

    def __str__(self) -> str:

        summary = (f"Root folder:\t{self.root}\n"
                    f"Files extracted in:\t{self.root.joinpath(tarball_root_name)}\n"
                    f"Task: {self.task}\n"
                    f"Line shape: {self.shape}\n"
                    f"Work folder:\t{self.work_folder_path}\n"
                    f"Data points:\t{len(self.data)}\n"
                    "Stats:\n"
                    f"{self.dataset_stats( self.data )}\n")
        if self.from_tsv_file:
             summary += "\nBuilt from TSV input:\t{}".format( self.from_tsv_file )
        if self.task == 'HTR':
            summary += "\nAlphabet:\t{} ({} codes)".format( self.work_folder_path.joinpath(alphabet_tsv_name), len(self.alphabet))
        
        prototype_alphabet = self.get_prototype_alphabet()

        summary += f"\n + A prototype alphabet generated from this subset would have {len(prototype_alphabet)} codes."
        
        symbols_shared = self.alphabet.symbol_intersection( prototype_alphabet )
        symbols_only_here, symbols_only_prototype = self.alphabet.symbol_differences( prototype_alphabet )


        summary += f"\n + Dataset alphabet shares {len(symbols_shared)} symbols with a data-generated charset."
        summary += f"\n + Dataset alphabet and a data-generated charset are identical: {self.alphabet == prototype_alphabet}"
        if symbols_only_here:
            summary += f"\n + Dataset alphabet's symbols that are not in a data-generated charset: {symbols_only_here}"
        if symbols_only_prototype:
            summary += f"\n + Data-generated charset's symbols that are not in the dataset alphabet: {symbols_only_prototype}"

        return ("\n________________________________\n"
                f"\n{summary}"
                "\n________________________________\n")


    def count_line_items(self, folder) -> Tuple[int, int]:
        """
        Count dataset items in the given folder.

        :param folder: 
            a directory path. 
        :type folder: str

        :returns: a pair `(<number of GT files>, <number of image files>)`
        :rtype: Tuple[int,int]

        """
        return (
                len( [ i for i in Path(folder).glob('*.gt.txt') ] ),
                len( [ i for i in Path(folder).glob('*.png') ] )
                )


    @staticmethod
    def get_default_alphabet() -> alphabet.Alphabet:
        """Return an instance of the default alphabet.


        :returns: alphabet.Alphabet: an alphabet instance.
        :rtype: alphabet.Alphabet

        """
        return alphabet.Alphabet( default_alphabet )


    def get_prototype_alphabet( self ) -> alphabet.Alphabet:
        """Return a prototype alphabet, generated from the transcriptions.

        :returns: alphabet.Alphabet: a prototypical alphabet instance, generated from the transcriptions.
        :rtype: alphabet.Alphabet

        """
        if self.data == []:
            logger.warning("Sample set is empty!")
            return None
        return alphabet.Alphabet( alphabet.Alphabet.prototype_from_data_samples( [ s['transcription'] for s in self.data ]))
            


class PadToWidth():
    """ Pad an image to desired length."""

    def __init__( self, max_w ):
        self.max_w = max_w

    def __call__(self, sample: dict) -> dict:
        """Transform a sample: only the image is modified, not the nominal height and width.
            
        """
        t_chw, h, w, gt = [ sample[k] for k in ('img', 'height', 'width', 'transcription') ]
        if w > self.max_w:
            warnings.warn("Cannot pad an image that is wider ({}) than the padding size ({})".format( w, self.max_w))
            return sample
        new_t_chw = torch.zeros( t_chw.shape[:2] + (self.max_w,))
        new_t_chw[:,:,:w] = t_chw

        # add a field
        mask = torch.zeros( new_t_chw.shape, dtype=torch.bool)
        mask[:,:,:w] = 1
        return {'img': new_t_chw, 'height': h, 'width': w, 'transcription': gt, 'mask': mask }




class ResizeToHeight():
    """Resize an image with fixed height, preserving aspect ratio as long as the resulting width
    does not exceed the specified max. width. If that is the case, the image is horizontally
    squeezed to fix this.

    """

    def __init__( self, target_height, max_width ):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, sample: dict) -> dict:
        """Transform a sample
            
           + resize 'img' value to desired height
           + modify 'height' and 'width' accordingly

        """
        t_chw, h, w, gt = [ sample[k] for k in ('img', 'height', 'width', 'transcription')]
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



def dummy():
    """ """
    return True
