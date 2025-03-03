#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import json
import warnings
import re
from typing import *


def segmentation_dict_from_xml(page: str) -> Dict[str,Union[str,List[Any]]]:
    """Given a pageXML file name, return a JSON dictionary describing the lines.
    Warning! Simplified logic assumes there each XML file describes exactly one
    page image - it is not always true (see Nuremberg dataset and related scripts for
    examples).

    Args:
        page (str): path of a PageXML file

    Returns:
        Dict[str,Union[str,List[Any]]]: a dictionary of the form::

            {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}

    """
    direction = {'0.0': 'horizontal-lr', '0.1': 'horizontal-rl', '1.0': 'vertical-td', '1.1': 'vertical-bu'}

    page_dict: Dict[str, Union['str', List[Any]]] = { 'type': 'baselines', 'text_direction': 'horizontal-lr' }

    def construct_line_entry(line: ET.Element, regions: list = [] ) -> dict:
            #print(regions)
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                return None
            bl_points = baseline_elt.get('points')
            if bl_points is None:
                return None
            baseline_points = [ [ int(p) for p in pt.split(',') ] for pt in bl_points.split(' ') ]
            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                return None
            c_points = coord_elt.get('points')
            if c_points is None:
                return None
            polygon_points = [ [ int(p) for p in pt.split(',') ] for pt in c_points.split(' ') ]

            transcription_elt, transcription_text = line.find('.//pc:Unicode', ns), ''
            transcription_text = ''.join( transcription_elt.itertext()) if transcription_elt is not None else ''

            return {'id': line_id, 'baseline': baseline_points, 
                    'boundary': polygon_points, 'regions': regions, 'text': transcription_text} 

    def process_region( region: ET.Element, line_accum: list, regions:list ):
        regions = regions + [ region.get('id') ]
        for elt in list(region.iter())[1:]:
            if elt.tag == "{{{}}}TextLine".format(ns['pc']):
                line_entry = construct_line_entry( elt, regions )
                if line_entry is not None:
                    line_accum.append( construct_line_entry( elt, regions ))
            elif elt.tag == "{{{}}}TextRegion".format(ns['pc']):
                process_region(elt, line_accum, regions)

    with open( page, 'r' ) as page_file:

        # extract namespace
        ns = {}
        for line in page_file:
            m = re.match(r'\s*<([^:]+:)?PcGts\s+xmlns(:[^=]+)?=[\'"]([^"]+)["\']', line)
            if m:
                ns['pc'] = m.group(3)
                page_file.seek(0)
                break

        if 'pc' not in ns:
            raise ValueError(f"Could not find a name space in file {page}. Parsing aborted.")

        lines = []

        page_tree = ET.parse( page_file )
        page_root = page_tree.getroot()

        pageElement = page_root.find('./pc:Page', ns)
        
        page_dict['imagename']=pageElement.get('imageFilename')
        page_dict['image_wh']=[ int(pageElement.get('imageWidth')), int(pageElement.get('imageHeight'))]
        
        for textRegionElement in pageElement.findall('./pc:TextRegion', ns):
            process_region( textRegionElement, lines, [] )

        page_dict['lines'] = lines

    return page_dict 
