"""
Interface to the Funsd dataset (words only)

Downloads and prepares the files needed for word OCR

"""

import torch
import torch.nn.functional as F
import pathlib as pl
import numpy as np
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from . import download_utils as du

import torchvision.datasets
from torchvision.datasets.vision import VisionDataset
import pathlib
import zipfile
import sys
import json
from . import lm_util
import uuid
import random
from typing import List, Tuple, Dict


class FunsdDataset(VisionDataset):
    """
    If you use this dataset for your research, please cite our paper:
    G. Jaume, H. K. Ekenel, J. Thiran "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents," 2019

    Bibtex format:
    @inproceedings{jaume2019,
        title = {FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
        author = {Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran},
        booktitle = {Accepted to ICDAR-OST},
        year = {2019}
    }


    Dataset format

    All the annotations are encoded in a JSON file. An example showing the annotations for the image below is presented. A detailed description of each entry from the JSON file is provided in the original paper.
    Img
            {
            "form": [
            {
                "id": 0,
                "text": "Registration No.",
                "box": [94,169,191,186],
                "linking": [
                    [0,1]
                ],
                "label": "question",
                "words": [
                    {
                        "text": "Registration",
                        "box": [94,169,168,186]
                    },
                    {
                        "text": "No.",
                        "box": [170,169,191,183]
                    }
                ]
            },
            ... 
        ]
        }
    """

    datafile = { 
                'url':  "https://guillaumejaume.github.io/FUNSD/dataset.zip",
                'filename': "dataset.zip",
                'md5': 'e05de47de238aa343bf55d8807d659a9',
                'desc': 'Forms images, with JSON meta-information', 
                'origin': 'public' 
            }

    base_folder='Funsd' # where files are extracted (relative to root)
    
    # subfolders 
    training_data_folder = "dataset/training_data"
    testing_data_folder = "dataset/testing_data" # to be splitted further into validation/testing subsets



    def __init__( 
                self, root: str,
                subset="train", 
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None, 
                image_loader=torchvision.datasets.folder.default_loader, 
                #min_gt_length=-1, max_gt_length=-1,
                extract=True,
                ) -> None:
        """
        Args:
            subset: subset to build, i.e. one of ["train" (default), "validate", "test"]
            extract: if False, assume that archives have already been extracted in the work directory
        """

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.debug = False

        self.root = root

        self.base_folder_path = pl.Path( root, self.base_folder )

        self.word_2_transcription = {}

        if not self.base_folder_path.exists() or not self.base_folder_path.is_dir():
            self.base_folder_path.mkdir(parents=True)

        self.download_and_extract( self.root, self.base_folder_path, self.datafile, extract )

        # Construct ground truth data and character encoding
        # Optional: write corresponding files (*.json and .tsv, respectively)

        # building training set (words)
        training_annotation_paths = [ p for p in self.base_folder_path.joinpath( self.training_data_folder, "annotations").iterdir() ]
        testing_annotation_paths = [ p for p in self.base_folder_path.joinpath( self.testing_data_folder, "annotations").iterdir() ]


        # same charset, no matter the set of items
        code_2_utf = self.get_charset( training_annotation_paths )
        code_2_utf.update( self.get_charset( testing_annotation_paths ))

        if subset == 'train':
            images_path = self.base_folder_path.joinpath( self.training_data_folder, "images")
            self.word_2_transcription = self.generate_word_items( images_path, training_annotation_paths )
        else:
            # split original test set in two
            validation_paths, testing_paths = self.split_set( testing_annotation_paths, 0.5 )
            images_path = self.base_folder_path.joinpath( self.testing_data_folder, "images")
            if subset == "test": 
                self.word_2_transcription = self.generate_word_items(images_path, testing_paths )
            elif subset == "validate":
                self.word_2_transcription = self.generate_word_items(images_path, validation_paths )


        # building encoder
        self.items = tuple( self.word_2_transcription.items() )
        print(f"Set = '{subset}' with {len(self.items)} words")

        self.encoder = lm_util.Encoder(code_2_utf=code_2_utf)

#        if min_gt_length>0:
#            if max_gt_length>=0:
#                self.file2transcriptions = {k:v for k,v in self.file2transcriptions.items() if len(v)>=min_gt_length and len(v)<=max_gt_length}
#            else:
#                self.file2transcriptions = {k: v for k, v in self.file2transcriptions.items() if len(v) >= min_gt_length}
#        elif max_gt_length>=0:
#                self.file2transcriptions = {k:v for k,v in self.file2transcriptions.items() if len(v)<=max_gt_length}

        #with open(self.base_folder_path.joinpath( "gt.json" ), "w", encoding="utf-8") as cmf:
        #    json.dump( self.word_2_transcription, cmf, indent=0)

        #with open(self.base_folder_path.joinpath("charset.tsv"), "w") as charset_file:
        #    print( self.encoder.get_tsv_string(), file=charset_file)

        if transform is None:
            self.transform = torchvision.transforms.ToTensor()

        self.image_loader = image_loader

    def get_sample_dictionary(self) -> List[Dict[str,str]]:
        """
        Return a sequence of pairs image/text (for Kraken).

        Returns: 
            A sequence where each sample is represented as a dictionary of the form::

            {'image': 'image_path', 'text': 'ground_truth_text'}.
        """
        return [ { 'image': k, 'text': v } for (k,v) in self.items ]


    def split_set(self, annotations_paths: list, cut: float) -> Tuple[tuple, tuple]:
        """
        Split a set of forms.
            annotations_paths: list of paths to XML form descriptions ('annotations')
            cut: (between 0 and 1) proportion of elements in the first set

        Returns:
            2 disjoint tuples of files paths
        """
        # use dict (=implicitly ordered) instead of sets to ensure deterministic
        # behaviour of random sampling
        print("Original set forms (testing) has {} elements".format( len(annotations_paths)))
        subset_count = int( cut * len(annotations_paths))

        
        random.seed(1)

        new_paths = random.sample( annotations_paths, len(annotations_paths)-subset_count )

        #original_set -= new_set
        annotations_paths = [ p for p in annotations_paths if p not in { p:None for p in new_paths } ]

        print("Subset #1 has {} forms; subset #2 has {} forms.".format( len(annotations_paths), len(new_paths)))

        return (tuple(annotations_paths), tuple(new_paths))

    def get_charset(self, annotations_subset: list) -> Dict[int, int]:
        
        charset = set()

        for annotation_item_path in annotations_subset:
            
            with open(annotation_item_path, "r") as annotation_file:
                json_annotation = json.load(annotation_file)

                for block in json_annotation['form']:
                    for word in block['words']:

                        if len(word['text']) > 0: 
                            for ch in word['text']:
                                charset.add( ch )
        return { idx: c for idx, c in enumerate(sorted(charset)) }


    def generate_word_items(self, images_path: pl.Path, annotations_subset: list ) -> Dict[str,str]:
        
        #data_path = pl.Path( self.root, self.base_folder_path, subset_folder_name )
        #annotations_path = data_path.joinpath("annotations")
        charset = set()
        image_2_transcription = {}

        #for annotation_item_path in annotations_path.iterdir():
        for annotation_item_path in annotations_subset:
            image_path = images_path.joinpath( annotation_item_path.name ).with_suffix(".png")
            
            with open(annotation_item_path, "r") as annotation_file, Image.open(image_path, "r") as form_img:
                json_annotation = json.load(annotation_file)

                for block in json_annotation['form']:
                    for word in block['words']:

                        if len(word['text']) > 0: 

                            word_img_raw = form_img.crop( word['box'] )
                            uniqid = str( uuid.uuid1()).split('-')[0]
                            image_path = self.base_folder_path.joinpath( uniqid + ".png" )
                            word_img_raw.save( image_path )

                            image_2_transcription[ str(image_path) ] = word['text']
        return image_2_transcription

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        filepath, transcription = self.items[item]
        pil_image = Image.new("L", (64,32), "white")
        try:
            #print(filepath)
            pil_image = self.image_loader(filepath)
        except:
                print(self.__dict__.keys())
                print("IMREAD failed")
        # PIL -> Tensor
        img = self.transform( pil_image )
        _, img_height, img_width = img.size()

        transcription = self.encoder.encode(transcription).reshape([1,-1]) # <-> transpose
        transcription_length = torch.LongTensor([ transcription.shape[1] ])
        transcription = torch.LongTensor( transcription )
        return ((img, img_width, img_height), (transcription, transcription_length))


    def download_and_extract(self, root: str, base_folder_path: pl.Path, fl_meta: dict, extract: bool=True) -> None:
        """
        TODO: factor out in utility module ??

        Args:
            root: where to save the archive
            base_folder: where to extract (any valid path)
            fl_meta: a dict with file meta-info (keys: url, filename, md5, origin, desc)
            extract: if False, skip the extraction step
        """
        output_file_path = pl.Path(root, fl_meta['filename'])
        print(output_file_path)
        if 'md5' not in fl_meta or not du.is_valid_archive(output_file_path, fl_meta['md5']):
            #gdown.download( fl_meta['url'], str(output_file_path), quiet=True, resume=True )
            du.resumable_download(fl_meta['url'], root, fl_meta['filename'], google=fl_meta['origin']=='google') 

        if not base_folder_path.exists() or not base_folder_path.is_dir():
            raise OSError("Base folder does not exist! Aborting.")

        with zipfile.ZipFile(output_file_path, 'r' ) as archive:
            print('Extract {} ({})'.format(output_file_path, fl_meta["desc"]))
            # exclude __MACOSX_* files
            keep = [ member for member in archive.namelist() if not member.startswith('__MACOSX') ]
            print(base_folder_path)
            archive.extractall( base_folder_path, keep)


def get_loader(dataset,batch_size=10,shuffle=True,num_workers=4):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=vgsl_collate)
    return data_loader



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

    captions, caption_lengths, img_list, widths, heights = zip(
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
    #print("by_length_idx={}".format(by_len_idx))

    #print(f"batch size = {batch.shape}")
    return (batch[by_len_idx,:,:,:], widths[by_len_idx], heights[by_len_idx]), (captions[by_len_idx,0,:], caption_lengths[by_len_idx])


if __name__ == "__main__":
    
    root = "."
    if len(sys.argv) > 1 and pl.Path(sys.argv[1]).is_dir():
        root = sys.argv[1]

    funsd_data = FunsdDataset(root, subset="train")
    funsd_data = FunsdDataset(root, subset="validate")
    funsd_data = FunsdDataset(root, subset="test")
    #data_loader = torch.utils.data.DataLoader( funsd_data, batch_size=4, shuffle=True, collate_fn=vgsl_collate )
    #print(list(data_loader))

