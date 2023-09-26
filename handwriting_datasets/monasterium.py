import sys
from typing import *
import defusedxml.ElementTree as ET
from pathlib import *
from PIL import Image, ImageDraw
from PIL import ImagePath as IP
import re
import argparse
import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms

torchvision.disable_beta_transforms_warning() # transforms.v2 namespaces are still Beta
from torchvision.transforms import v2


"""
Generate image/GT pairs for Monasterium transcriptions

1. Use the metadata files (in ./mets) to map PageXML files (in ./page_urls) with the images they describe (in ./page_imgs)
2. Extract line polygons from each image and their respective transcriptions from the PageXML file

"""


class MonasteriumDataset(VisionDataset):

    def __init__(
                self,
                basefolder: str,
                subset: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract: bool = True,
                target_folder: str ='line_imgs',
                limit: int = 0
                ):

        tf = v2.PILToTensor()
        if transform:
            tf = v2.Compose( [ v2.PILToTensor(), transform ] )


        super().__init__(basefolder, transform=tf, target_transform=target_transform )
        self.setname = 'Monasterium'
        self.basefolder=basefolder

        self.pagexmls = Path(self.basefolder).joinpath('page_urls').glob('*.xml')
        #self.target_folder = Path('.', target_folder)

        self.data = []

        if not extract:
            return

        self.data = self.extract_lines( target_folder, limit=limit ) 

    def purge(self, folder: str) -> int:
        cnt = 0
        for line in Path( folder ).iterdir():
            line.unlink()
            cnt += 1
        return cnt

    def map_pagexml_to_img_id(self) -> dict:
        """
        Maps each PageXML to its document image's Id

        Returns:

            (dict, dict): two dictionaries, that respectively map the PageXML ids to the document
                            image ids and conversely.
        """
        img2xml = {}
        xml2img = {}

        mets = Path(self.basefolder).joinpath('mets').glob('*.xml')

        for met in mets:
            with open(met, 'r') as met_file:
                met_tree = ET.parse( met_file )
                ns = {  'ns2': "http://www.w3.org/1999/xlink",
                        'ns3': "http://www.loc.gov/METS/" }

                met_root = met_tree.getroot()

                url_re = re.compile(r'.*id=([A-Z]+)&?.*')

                img_grp = met_root.find(".//ns3:fileGrp[@ID='IMG']", ns)
                xml_grp = met_root.find(".//ns3:fileGrp[@ID='PAGEXML']", ns)

                img_ids=[]
                xml_ids=[]

                for img_tag in img_grp.findall(".//ns3:FLocat", ns):
                    url = img_tag.get(f"{{{ns['ns2']}}}href")
                    img_ids.append( url_re.sub(r'\1', url) )


                for xml_tag in xml_grp.findall(".//ns3:FLocat", ns):
                    url = xml_tag.get(f"{{{ns['ns2']}}}href")
                    xml_ids.append( url_re.sub(r'\1', url) )

                for img, xml in zip(img_ids, xml_ids):
                    img2xml[img]=xml
                    xml2img[xml]=img

        return xml2img


    def extract_lines(self, target_folder: str, shape='polygons', text_only=False, limit=0) -> List[Tuple[str, str]]:
        """
        Generate line images from the PageXML files
        
        - images are saved in the local directory of the consumer's program
        - generates a CSV file with all pairs (img_file_path, transcription)
        
        Returns:
            list: an array of pairs (img_file_path, transcription)

        """
        Path( target_folder ).mkdir(exist_ok=True) # always create the subfolder if not already there
        self.purge( target_folder ) # ensure there are no pre-existing line items in the target directory

        gt_lengths = []
        img_sizes = []
        count = 0 # for testing purpose

        xml2img = self.map_pagexml_to_img_id()

        items = []

        for page in self.pagexmls:

            xml_id = Path( page ).stem
            if xml_id not in xml2img:
                continue
            image_id = xml2img[ xml_id ]
            img_path = Path(self.basefolder, 'page_imgs', f'{image_id}.jpg')

            with open(page, 'r') as page_file:

                page_tree = ET.parse( page_file )

                ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

                page_root = page_tree.getroot()

                if not text_only:
                    page_image = Image.open( img_path, 'r')

                for textline_elt in page_root.findall( './/pc:TextLine', ns ):
                    if limit and count == limit:
                        return items

                    textline = dict()
                    textline['id']=textline_elt.get("id")
                    textline['transcription']=textline_elt.find('./pc:TextEquiv', ns).find('./pc:Unicode', ns).text
                    polygon_string=textline_elt.find('./pc:Coords', ns).get('points')
                    coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
                    textline['bbox'] = IP.Path( coordinates ).getbbox()
                    img_sizes.append( [ textline['bbox'][i+2]-textline['bbox'][i] for i in (0,1) ])
                    img_path_prefix = Path(target_folder, image_id + "-" + textline['id'] )
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
        print(items)
        return items


    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            tuple: a pair (image, transcription)
        """
        transcr = self.data[index][1]
        img = self.transform( Image.open(self.data[index][0], 'r') )
        return (img, transcr)


    def __len__(self) -> int:
        """
        Returns:
            tuple: a pair (image, transcription)
        """
        return len( self.data )





    def count_line_items(self, folder) -> Tuple[int, int]:
        return (
                len( [ i for i in Path(folder).glob('*.gt') ] ),
                len( [ i for i in Path(folder).glob('*.png') ] )
                )

#
#print("--------- Img sizes")
#img_sizes_array = np.array( img_sizes )
#
#means = img_sizes_array.mean(axis=0)
#maxs = img_sizes_array.max(axis=0)
#mins = img_sizes_array.min(axis=0)
#medians = np.median(img_sizes_array, axis=0)
#
#print('Width: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means[0], mins[0], maxs[0], medians[0]))
#print('Height: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means[1],mins[1], maxs[1], medians[1]))
#
#
#print("\n-------- GT lengths")
#gt_lengths_array = np.array( gt_lengths)
#
#means = gt_lengths_array.mean()
#maxs = gt_lengths_array.max()
#mins = gt_lengths_array.min()
#medians = np.median(gt_lengths_array)
#
#print('Width: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means, mins, maxs, medians))
#
#
