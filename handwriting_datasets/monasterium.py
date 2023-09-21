import sys
import glob
import defusedxml.ElementTree as ET
from pathlib import *
from PIL import Image, ImageDraw
from PIL import ImagePath as IP
import re
import argparse
import numpy as np


"""
Generate image/GT pairs for Monasterium transcriptions

1. Use the metadata files (in ./mets) to map PageXML files (in ./page_urls) with the images they describe (in ./page_imgs)
2. Extract line polygons from each image and their respective transcriptions from the PageXML file

"""

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--text_only", action="store_true", help="Do not extract the line iamges")

args = parser.parse_args()


pagexmls = glob.glob("./page_urls/*.xml")
mets = glob.glob("./mets/*xml")

img_to_xml = {}
xml_to_img = {}

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
            img_to_xml[img]=xml
            xml_to_img[xml]=img

gt_lengths = []
img_sizes = []

for page in pagexmls:

    xml_id = Path( page ).stem
    if xml_id not in xml_to_img:
        continue
    image_id = xml_to_img[ xml_id ]
    img_path = Path('page_imgs', f'{image_id}.jpg')

    with open(page, 'r') as page_file:

        page_tree = ET.parse( page_file )

        ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

        page_root = page_tree.getroot()

        if not args.text_only:
            print(img_path)
            page_image = Image.open( img_path, 'r')

        for textline_elt in page_root.findall( './/pc:TextLine', ns ):
            textline = dict()
            textline['id']=textline_elt.get("id")
            textline['transcription']=textline_elt.find('./pc:TextEquiv', ns).find('./pc:Unicode', ns).text
            polygon_string=textline_elt.find('./pc:Coords', ns).get('points')
            coordinates = [ tuple(map(int, pt.split(','))) for pt in polygon_string.split(' ') ]
            textline['bbox'] = IP.Path( coordinates ).getbbox()
            img_sizes.append( [ textline['bbox'][i+2]-textline['bbox'][i] for i in (0,1) ])
            textline['img_path']=image_id + "-" + textline['id']

            if not args.text_only:
                bbox_img = page_image.crop( textline['bbox'] )

                # mask (=line polygon)
                mask = Image.new("L", bbox_img.size, 0)
                drawer = ImageDraw.Draw(mask)
                leftx, topy = textline['bbox'][:2]
                transposed_coordinates = [ (x-leftx, y-topy) for x,y in coordinates ]
                drawer.polygon( transposed_coordinates, fill=255 )

                # background
                bg = Image.new("RGB", bbox_img.size, color=0)

                # composite
                Image.composite(bbox_img, bg, mask).save( textline['img_path'] + '.png')

            with open(textline['img_path']+'.gt', 'w') as gt_file:
                gt_file.write( textline['transcription'] )
                gt_lengths.append(len( textline['transcription']))


print("--------- Img sizes")
img_sizes_array = np.array( img_sizes )

means = img_sizes_array.mean(axis=0)
maxs = img_sizes_array.max(axis=0)
mins = img_sizes_array.min(axis=0)
medians = np.median(img_sizes_array, axis=0)

print('Width: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means[0], mins[0], maxs[0], medians[0]))
print('Height: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means[1],mins[1], maxs[1], medians[1]))


print("\n-------- GT lengths")
gt_lengths_array = np.array( gt_lengths)

means = gt_lengths_array.mean()
maxs = gt_lengths_array.max()
mins = gt_lengths_array.min()
medians = np.median(gt_lengths_array)

print('Width: avg={:.1f}, min={:.1f}, max={:.1f}, median={:.1f}'.format(means, mins, maxs, medians))


