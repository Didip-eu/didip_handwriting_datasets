"""
Interface to the IAM dataset (text lines only)

Downloads and prepares the files needed for the text line recognition task.

"""

import torchvision
import torch
from torch.utils.data import Dataset

import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import defusedxml.ElementTree as ET
from . import download_utils as du

import torchvision.datasets
from torchvision.datasets.vision import VisionDataset
import pathlib
import tarfile
import zipfile
import sys
import json
from . import lm_util
import argparse
from typing import List, Dict, Tuple




files = [
        {   'url': "https://drive.google.com/uc?export=download&id=1iz9BcpivWMrvoskAyZG48EbDqDPuzyRC",
            'filename': "largeWriterIndependentTextLineRecognitionTask.zip",
            'md5': 'e1707fc9ed31550f1cbb61e3bff4df52',
            'desc': 'List of images in training and validation sets for line recognition task (22K)',
            'origin': 'google'

        },
        {   'url': "https://drive.google.com/uc?export=download&id=1s2GBlEqUaziwj6RBhCn3R10lxdc5ug1X",
            'filename': "xml.tgz",
            'md5': 'f041f4b062e1fb27a09632d2bb921dfd',
            'desc': 'Meta-data (one XML file per form)',
            'origin': 'google'

        },
        {   'url': "https://drive.google.com/uc?export=download&id=11LdQK-JOi5-dU0Hnj64a2BLyG4T05f8u",
            'filename': "lines.tgz",
            'md5': '100530f7aac0f9a670bb2c21786cce70',
            'desc': 'Lines images (638M)',
            'origin': 'google'
        },
        {   'url': "https://drive.google.com/uc?export=download&id=1w0WQQ-6lWRJ6gADzCetbvuY150PWpi7T",
            'filename': "words.tgz",
            'md5': '58b1e255e455f45e08e3e1029a8e5984',
            'desc': 'Word images (783M)',
            'origin': 'google'
        },
        {   'url': "https://drive.google.com/uc?export=download&id=17W7y9K4MC7q3_EU_tCSYyJw4Y2R3Hm4n",
            'filename': "ascii.tgz",
            'md5': 'c5b0914b766de2e405f9f8be86d0a662',
            'desc': 'Meta-data on forms, lines, and words, including transcriptions (2.5M, tabulated format)',
            'origin': 'google'
        }
]


# Where files are extracted: just under the root and is no meant to be modified
base_folder_name='IAMHandwritingDataset' 
work_folder_name='IAMHandwritingDatasetWork' 

class IAMVisionDataset(VisionDataset):
    """
    IAM Handwriting dataset, from
        URL: https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
    Access to the files requires an authentication step, hence the alternate download location (Google drive).

    + builds the dataset described in the Line Recognition Task provided by the authors
    + TODO:
      - factor out utility functions

    """

    def __init__(
                self, root: str=str(Path.home().joinpath('tmp', 'data', 'IAM')),
                work_folder: str = '', # here further files are created, for any particular task
                subset: str ="train",
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                image_loader=torchvision.datasets.folder.default_loader,
                min_gt_length: int = -1, max_gt_length: int = -1,
                task: str = "lines",
                limit: int = 0,
                extract=True) -> None:
        """
        Goal: Initialize a ready-to-use loader, with default: lines (not words), trainset


        Args:
            root: location where base folder is created (all tarballs are downloaded in the root,
                  then extracted in the base folder below) - default: current/project directory
            subset: one of ["train" (default), "validate1", "validate2", "test"]
            task: "lines" (as provided by the authors), or "words" (derived from the line sets, for experimental purpose)
            limit: construct a mini-dataset of lines, with indicated number of lines (for testing purpose)
            extract: if False, assume that archives have already been extracted in the work directory
        """

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.debug = False

        self.base_folder_path = Path( root, base_folder_name )
        self.work_folder_path = Path( root, work_folder_name ) if work_folder == '' else Path( work_folder )

        for folder in [ self.base_folder_path, self.work_folder_path ]:
            if not folder.exists() or not folder.is_dir():
                folder.mkdir( parents=True )

        # note: self.root is a VisionDataset attribute
        for fl in files:
            self.download_and_extract( self.root, self.base_folder_path, fl, extract)

        # Line-to-transcription mapping
        # E.g.
        # a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
        # a01-000u-01 ok 156 19 395 932 1850 105 nominating|any|more|Labour|life|Peers
        line_descriptions = "lines.txt"

        # Word-to-transcription mapping
        # E.g.
        # a01-000u-00-00 ok 154 408 768 27 51 AT A
        # a01-000u-00-01 ok 154 507 766 213 48 NN MOVE
        word_descriptions = "words.txt"

        # Subset lists for predefined task (Line Recognition Task)
        # Image for line a01-000u-00 is in a01/a01-000u/a01-000u-01.png
        # List of training samples (lines), in the form <form>-<mid>-<line>
        # E.g.
        # a01-000u-00
        # a01-000u-01
        # ...
        subset_2_input_file = {
                'train': 'trainset.txt',
                'validation1': 'validationset1.txt',
                'validation2': 'validationset2.txt',
                'test': 'testset.txt' }


        self.object_2_transcription = {}
        code_2_utf = {}

        if task == "lines" or limit:
            # create dictionary {<line id>: transcription}
            self.object_2_transcription, code_2_utf = self.line_to_transcription( self.base_folder_path.joinpath( line_descriptions ))
            # input file of line ids is used without further ado
            input_list = self.list_from_file(
                            Path(self.base_folder_path).joinpath( subset_2_input_file[subset] ), limit)

        elif task == "words":
            # create dictionary {<word id>: transcription}
            self.object_2_transcription, code_2_utf = self.word_to_transcription( self.base_folder_path.joinpath( word_descriptions ))
            # input file of line ids is used to derive a list of words
            input_list = self.generate_word_set(
                    self.base_folder_path,
                    self.base_folder_path.joinpath( subset_2_input_file[subset] ))

        # In case length of GT texts is constrained 
        if min_gt_length>0:
            if max_gt_length>=0:
                self.object_2_transcriptions = {k:v for k,v in self.object_2_transcriptions.items() if len(v)>=min_gt_length and len(v)<=max_gt_length}
            else:
                self.object_2_transcriptions = {k: v for k, v in self.object_2_transcriptions.items() if len(v) >= min_gt_length}
        elif max_gt_length>=0:
                self.object_2_transcriptions = {k:v for k,v in self.object_2_transcriptions.items() if len(v)<=max_gt_length}

        # convert {<line or word id>: transcription} →  dictionary { <object image path>: transcription } 
        image_2_transcription = self.object_image_to_transcription( input_list )
        #print(len(image_2_transcription.keys()))


        # CRITICAL LINE: building the encoder 
        self.items = tuple( image_2_transcription.items() )
        self.encoder = lm_util.Encoder(code_2_utf=code_2_utf)

        print(f'Encoder: {len(self.encoder)} entries')



        # Groundtruth and charset files: for debugging purpose only (not needed for training)
        # with open(self.base_folder_path.joinpath( "gt.json" ), "w", encoding="utf-8") as cmf:
        #    json.dump( image_2_transcription, cmf, indent=0)

        with open(self.work_folder_path.joinpath("charset.tsv"), "w") as charset_file:
            print( self.encoder.get_tsv_string(), file=charset_file)

        if transform is None:
            self.transform = lambda x: x

        self.image_loader = image_loader


    def __len__(self):
        return len(self.items)


    def __getitem__(self, item):
        filepath, transcription = self.items[item]
        default_image = Image.new("RGB", (64,32), "white")
        pil_image = None
        try:
            pil_image = self.image_loader(filepath)
        except:
                print(filepath)
                print(self.__dict__.keys())
                print("IMREAD failed (using default image)")
                pil_image = default_image
        # default: PIL -> Tensor
        img = torchvision.transforms.ToTensor()( self.transform( pil_image ) )
        _, img_height, img_width = img.size()

        # Text transform
        if self.target_transform:
            transcription = self.target_transform( transcription )

        transcription = self.encoder.encode(transcription).reshape([1,-1]) # <-> transpose
        transcription_length = torch.LongTensor([ transcription.shape[1] ])
        transcription = torch.LongTensor( transcription )
        return ((img, img_width, img_height), (transcription, transcription_length))


    def extract_image_and_text_file(self, subfolder: str, dummy = False) -> int:
        """
        For Kraken only: extract image/text pairs in a single directory where each pair
        shares a prefix, with "*.gt.txt" files containing the transcritions; apply any
        transform defined at initialization time.

        Args:
            subfolder (str): subfolder in which image and GT files are to be created.
            dummy (bool): do not create the pairs, assuming they are already there (for ease of debugging)

        Returns:
            int: number of items in the dataset
        """
        kraken_gt_folder = Path( self.root,  subfolder)
        kraken_gt_folder.mkdir(exist_ok=True)
        if dummy:
                return len(self.items)
        for file in kraken_gt_folder.glob("*"):
            file.unlink()

        for (k,v) in self.items:
            image_file = Path(k)
            basename = image_file.name
            local_link = Path(kraken_gt_folder, basename)
            local_link.unlink(missing_ok=True)
            # should apply image transform here
            #local_link.symlink_to( Path("..", image_file ))

            transformed_img = self.transform( Image.open( image_file ))
            transformed_img.save( local_link )

            with open( local_link.with_suffix(".gt.txt"), "w") as gt_file:
                gt_file.write( self.target_transform(v) if self.target_transform else v )

        return len(self.items)


    def generate_word_set(self, base_folder_path: Path, lineset_file_path: Path) -> list:
        """
        Generate word lists for training or testing: word lists are derived from the line recognition task sets
        provided by the authors.

        Args:
            line_file_path: the list of lines from which words should be taken

        """
        print(f'Building word set from {lineset_file_path}...')
        lines_to_keep = set()
        word_set = set()
        writer_set = set()
        with open( lineset_file_path, "r" ) as lineset_file:
            for line in lineset_file:
                lines_to_keep.add( line[:-1] )
        for w, metadata in self.get_word_metadata( base_folder_path ).items():
            # skip on empty transcriptions
            if metadata[0] in lines_to_keep:
                word_set.add( w )
                writer_set.add( metadata[1]) # for reference only
        print('[Derived from {}] - words: {} writers: {}'.format(lineset_file_path.name, len(word_set), len(writer_set)))
        return list(word_set)



    def get_sample_dictionary(self) -> List[Dict[str,str]]:
        """
        Return a sequence of pairs <image path>/text (for Kraken).

        Returns:
            A sequence where each sample is represented as a dictionary of the form::

            {'image': 'image_path', 'text': 'ground_truth_text'}.
        """
        return [ { 'image': k, 'text': self.target_transform(v) if self.target_transform else v, 'preparse': True} for (k,v) in self.items ]



    def get_word_metadata(self, base_folder_path: Path) -> dict:
        """
        Maps words to their line id and writer.

        Args:
            base_folder_path: location of XML metadata files (one per form)

        Returns:
            A dictionary of the form { word: (line, writer) }
        """
        # mapping words to meta-data (line, writer)
        word_2_metadata = {}

        for fl in base_folder_path.iterdir():
            if fl.suffix == '.xml':
                with open( fl ) as form_desc:
                    try:
                        form_tree = ET.parse( form_desc )
                        form_root = form_tree.getroot()
                        writer = form_root.get('writer-id')
                        for line_elt in form_root.iter('line'):
                            line = line_elt.get('id')
                            for word_elt in line_elt.findall('word'):
                                word = word_elt.get('id')
                                word_2_metadata[ word ] = (line, writer)
                    except ET.ParseError as e:
                        print(e)
        return word_2_metadata


    def list_from_file(self, list_file_path: Path, limit: int = 0) -> list:
        """
        Make a list of line ids (=file prefixes for the images).

        Args:
            list_file_path: file containing the line ids (one per line). Eg.::

            a01-049x-07
            a01-049x-08
            ...

           limit: allows for generating mini-datasets, for debugging purpose
        """
        input_list = []
        with open(list_file_path, "r") as input_list_file:
            for lno, l in enumerate(input_list_file, 1):
                if limit and lno > limit:
                    break
                if l:
                    input_list.append( l[:-1] )
        return input_list


    def line_to_transcription( self, file_path: Path) -> Tuple[dict, dict]:
        """
        Map line ids to their transcriptions and chars to integer codes.

        Args:
            file_path: meta-data for all lines, in the form:
                       01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from


        Returns:
            tuple: a pair (<mapping of image id to transcription>, <mapping of integer to character>)
        """
        if not file_path.exists():
            print(f"File with line descriptions {file_path} not found! Image-transcription mapping aborted.")
            return {}
        l2t = {}
        chars = set()
        with open( file_path, "r") as line_descriptions:
            for line_desc in line_descriptions:
                if line_desc[0]=='#':
                    continue
                line_properties = line_desc[:-1].split(' ')
                line_id, line_transcription_sequence = line_properties[0], line_properties[-1]
                l2t[line_id] = line_transcription_sequence.replace('|', ' ')
                for ch in l2t[line_id]:
                    chars.add(ch)
        code_2_utf = { idx: c for idx, c in enumerate(sorted(chars)) }
        return (l2t, code_2_utf)


    def word_to_transcription( self, file_path: Path ) -> Tuple[dict, dict]:
        """
        Map word ids to their transcriptions and chars to integer codes.

        Args:
            file_path: meta-data for all words, in the form:
                       01-014-00-06 ok 182 1941 719 424 111 VBG boycotting

        Returns:
            tuple: a pair (<mapping of word id to transcription>, <mapping of integer to character>)
        """
        if not file_path.exists():
            print(f"File with word descriptions {file_path} not found! Image-transcription mapping aborted.")
            return {}
        l2t = {}
        chars = set()
        with open( file_path, "r") as word_descriptions:
            for word_desc in word_descriptions:
                if word_desc[0]=='#':
                    continue
                word_properties = word_desc[:-1].split(' ')
                word_id, word_transcription_sequence = word_properties[0], word_properties[-1]
                l2t[word_id] = word_transcription_sequence
                for ch in l2t[word_id]:
                    chars.add(ch)
        code_2_utf = { idx: c for idx, c in enumerate(sorted(chars)) }
        return (l2t, code_2_utf)


    def object_image_to_transcription(self, object_list: list ) -> Dict[str, str]:
        """
        Map object image paths to their transcriptions.

        Args:
            object_list: list of image ids

        Returns:
            dict: mapping of image path to transcription
        """
        i2t = {}
        for fl in object_list:
            form, mid = fl.split('-')[:2]
            # path constructed from image/line id
            # Ex.
            # Line 'h05-012-00' and word 'h05-012-00-00' are both stored in h05/h05-012,
            # in h05-012-00.png and 05-012-00-00.png, respectively
            object_image_path = Path(self.base_folder_path, form, form + "-" + mid, fl + ".png")
            object_transcription = self.object_2_transcription[fl] if fl in self.object_2_transcription else None
            # filter out empty images
            if object_image_path.stat().st_size > 0 and object_transcription:
                i2t[ str(object_image_path) ] = object_transcription
        return i2t


    def download_and_extract(
            self, root: str, 
            base_folder_path: Path, 
            fl_meta: dict, extract=True) -> None:
        """
        Download the archive and extract it. If a valid archive already exists in the root location,
        extract only; Boolean flag 'extract' is False, just check that the base folder does exist and
        return.

        TODO: factor out in utility module ??

        Args:
            root: where to save the archive
            base_folder: where to extract (any valid path)
            fl_meta: a dict with file meta-info (keys: url, filename, md5, origin, desc)
        """
        output_file_path = Path(root, fl_meta['filename'])

        #print( fl_meta, type(fl_meta['md5']) )
        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            #gdown.download( fl_meta['url'], str(output_file_path), quiet=True, resume=True )
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


def vgsl_collate(batch_list: list) -> tuple:
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (1|3, N, M); variable width height.
            - caption: torch integer tensor of shape (K); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 1|3, max(N), max(M)).
        widths: torch integer tensor of shape (batch_size)
        heights: torch integer tensor of shape (batch_size)
        caption: torch tensor of shape (batch_size, max(K)).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    batch_list.sort(key=lambda x: len(x[1]), reverse=True)

    captions, caption_lengths, img_list, widths,heights = zip(
        *[(item[1][0], item[1][1], item[0][0], item[0][1], item[0][2]) for item in batch_list])

    caption_lengths = torch.stack(caption_lengths,dim=0).view(-1)
    widths = torch.LongTensor(widths)
    heights = torch.LongTensor(heights)

    max_caption_length = torch.max(caption_lengths)
    max_width = torch.max(widths)
    max_height = torch.max(heights)

    pad_img = lambda x: F.pad(x, (0, max_width - x.size(2), 0, max_height - x.size(1), 0, 0))
    pad_caption = lambda x: F.pad(x, (0, max_caption_length - x.size(1)))

    batch = torch.stack([pad_img(img) for img in img_list])

    captions = torch.stack([pad_caption(caption) for caption in captions])

    widths=torch.LongTensor(widths)
    heights=torch.LongTensor(heights)
    caption_lengths=torch.LongTensor(caption_lengths)

    by_len_idx=widths.sort(descending=True)[1]

    return (batch[by_len_idx,:,:,:], widths[by_len_idx], heights[by_len_idx]), (captions[by_len_idx,0,:], caption_lengths[by_len_idx])


def get_loader(dataset,batch_size=10,shuffle=True,num_workers=4):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=vgsl_collate)
    return data_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", help="Root directory", default=".", type=str)
    parser.add_argument("-t", "--task", help="Task", choices=['lines', 'words'], default="lines", type=str)
    parser.add_argument("-s", "--subset", help="data subset", choices=['train', 'validation1', 'validation2'], default="train", type=str)
    args = parser.parse_args()

    iam_data = IAMDataset(args.root, subset=args.subset, task=args.task, extract=False)
    data_loader = torch.utils.data.DataLoader( iam_data, batch_size=4, shuffle=True, collate_fn=vgsl_collate )
    #print(list(data_loader))

