#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import argparse 

from didip_handwriting_datasets import xml_utils as xu


"""
Convert PageXML segmentation files to their JSON equivalent.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+', help='Input files')
    parser.add_argument('--out', choices=['json', 'stdout'], default='stdout')


    args = parser.parse_args()
    print(args)

    for file_name in args.files:
        page = Path( file_name )
        if page.suffix != '.xml':
            continue
        if args.out == 'json':
            outfile_name = page.with_suffix('.json')
            with open(page, 'r') as infile, open( outfile_name, 'w') as outfile:
                print( json.dumps( xu.pagexml_to_segmentation_dict( page ) ), file=outfile)
                print(f'{page} → {outfile_name}')
        elif args.out == 'stdout':
            print( json.dumps( xu.pagexml_to_segmentation_dict( page ) ))



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
            print( json.dumps(xu.pagexml_to_segmentation_dict( page )), file=of)
            count += 1
    print(f"Converted {count} PageXMl files to JSON")


