#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import argparse 
import xml.etree.ElementTree as ET



def pagexml_to_segmentation_dict(page: str) -> dict:
    """
    Given a pageXML file, return a JSON dictionary describing the lines.

    Args:
        page (str): path of a PageXML file
    Output:
        dict: a dictionary of the form

        {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}

    """
    direction = {'0.0': 'horizontal-lr', '0.1': 'horizontal-rl', '1.0': 'vertical-td', '1.1': 'bu'}

    with open( page ) as page_file:
        page_tree = ET.parse( page_file )
        ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        page_root = page_tree.getroot()

        page_elt = page_root.find('./pc:Page', ns)
        if page_elt is not None:
            image_filename, width, height = [ page_elt.get( attr ) for attr in ('imageFilename', 'imageWidth', 'imageHeight') ]
        else:
            sys.exit()

        page_dict = { 
                'imagename': image_filename,
                'image_wh': [ int(width), int(height) ],
                'type': 'baselines',
        }

        #page_dict['text_direction'] = direction[ page_root.find('.//pc:TextRegion', ns).get( 'orientation' )]

        lines_object = []
        for line in page_root.findall('.//pc:TextLine', ns):
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                continue
            baseline_points = [ [ int(p) for p in pt.split(',') ] for pt in baseline_elt.get('points').split(' ') ]

            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                continue
            polygon_points = [ [ int(p) for p in pt.split(',') ] for pt in coord_elt.get('points').split(' ') ]

            transcription_elt = line.find('.//pc:Unicode', ns)
            transcription_text = ''
            if transcription_elt is not None:
                transcription_text = ''.join( transcription_elt.itertext())

            lines_object.append( {'id': line_id, 'baseline': baseline_points, 'boundary': polygon_points, 'text': transcription_text} )

        page_dict['lines'] = lines_object

    return page_dict 





"""
Convert PageXML segmentation files to their JSON equivalent.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+', help='Input files')
    parser.add_argument('--out', choices=['json', 'stdout'], default='stdout')


    args = parser.parse_args()

    for file_name in args.files:
        page = Path( file_name )
        if page.suffix != '.xml':
            continue
        if args.out == 'json':
            outfile_name = page.with_suffix('.json')
            with open(page, 'r') as infile, open( outfile_name, 'w') as outfile:
                print( json.dumps( pagexml_to_segmentation_dict( page ) ), file=outfile)
                print(f'{page} → {outfile_name}')
        elif args.out == 'stdout':
            print(json.dumps( pagexml_to_segmentation_dict( page ) ))



def pages_to_json(dir_path: str):
    """
    Convert a full dir of PageXML segmentation files to their JSON equivalent.

    Args:
        dir_path (str): a directory containing PageXML segmentation data
    """

    count=0
    for page in Path( dir_path ).glob('*.xml'):
        print( page )
        with open( page.with_suffix('.json'), 'w') as of:
            print( json.dumps(pagexml_to_segmentation_dict( page )), file=of)
            count += 1
    print(f"Converted {count} PageXMl files to JSON")


