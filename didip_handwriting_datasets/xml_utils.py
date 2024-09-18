import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import json
import warnings



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

        page_dict = { 'type': 'baselines' }

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

            lines_object.append( {'line_id': line_id, 'baseline': baseline_points, 'boundary': polygon_points} )

        page_dict['lines'] = lines_object

    return page_dict 



