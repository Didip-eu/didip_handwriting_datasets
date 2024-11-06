#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import re


USAGE=f"USAGE: {sys.argv[0]} <tei.xml> [ <outfile.xml> ]"

def tei_path_to_line_dict( tei_path: str ) -> dict:
    with open( tei_path, 'r') as tei:

        tei_tree = ET.parse( tei )
        ns = { 'pc': "http://www.tei-c.org/ns/1.0" }

        tei_root = tei_tree.getroot()

        body = tei_root.find('.//pc:text/pc:body', ns)

        page_source_image = ''
        line_dict = {}
        for elt in body.iter():
            if elt.tag == "{{{}}}pb".format( ns['pc'] ):
                page_source_image = elt.get('source') 
                line_dict[page_source_image]={}
                print( page_source_image)
            elif elt.tag == "{{{}}}p".format( ns['pc'] ):
                line_id = ''
                text = ''
                for page_atom in list(elt.iter())[1:]:
                    if page_atom.tag == "{{{}}}lb".format(ns['pc']):
                        text = ''
                        line_id = re.sub(r'^.+_(line_.+)', r'\1', page_atom.get('facs'))
                    elif page_atom.tag == "{{{}}}choice".format(ns['pc']):
                        text += page_atom.find('./pc:expan',ns).text
                    text += page_atom.tail if page_atom.tail is not None else ''
                    if text != '':
                        line_dict[page_source_image][line_id]=re.sub(r'\s+', r' ', text.strip())
        return line_dict 


def page_xml_expand_text( page_path: str, line_dict ) -> dict:
    with open( page_path, 'r') as tei:

        page_tree = ET.parse( page_path )
        ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" }

        page_root = page_tree.getroot()
        
        for textRegionElt in page_root.findall('.//pc:TextRegion', ns):
            region_text = []
            for textLineElt in textRegionElt.findall('./pc:TextLine', ns):

                line_id = textLineElt.get('id')
                if line_id in line_dict:
                    new_transcription = line_dict[ line_id ]
                    textLineElt.find('.//pc:TextEquiv/pc:Unicode', ns).text=new_transcription 
                    region_text.append( new_transcription )

            textRegionElt.find('./pc:TextEquiv/pc:Unicode', ns).text = '\n'.join( region_text )


        return page_tree



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit()


    line_dict = tei_path_to_line_dict( sys.argv[1] )

    print(list(sorted(line_dict[lkd].keys()) for lkd in line_dict.keys()))

    for page_stem in line_dict.keys():

        if not Path(f'{page_stem}.xml').exists():
            continue
        new_tree = page_xml_expand_text( page_stem + '.xml', line_dict[ page_stem ]  )

        if len(sys.argv) > 2:
            with open( sys.argv[2], 'w') as of:
                new_tree.write(f"{page_stem}_mod.xml", encoding='utf-8')
        else:
            new_tree.write(sys.stdout.buffer, encoding='utf-8')



                
