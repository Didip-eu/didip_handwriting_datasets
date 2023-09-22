import unittest
from handwriting_datasets.monasterium import MonasteriumDataset

class MonasteriumTest( unittest.TestCase ):

    def test_dataset_dummy(self):
        monasterium_data = MonasteriumDataset('.', subset='train', extract=False)
        self.assertTrue(True)

    
    def test_pagexml_to_img_id_mapping(self):
        bf = '/home/nicolas/htr/Monasterium'
        xml2id = MonasteriumDataset(bf, subset='train', extract=False).map_pagexml_to_img_id()
        self.assertTrue( len(xml2id) )

    def test_pagexml_to_img_line_extraction(self):
        bf = '/home/nicolas/htr/Monasterium'
        ms = MonasteriumDataset(bf, subset='train', extract=False)
        ms.extract_lines(shape='bbox')
        self.assertTrue( ms.has_line_img() )

    

    

if __name__ == "__main__":
    unittest.main() 

