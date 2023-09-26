import unittest
from handwriting_datasets.monasterium import MonasteriumDataset
from torch import Tensor
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

class MonasteriumTest( unittest.TestCase ):

    def test_dataset_dummy(self):
        monasterium_data = MonasteriumDataset('.', subset='train', extract=False)
        self.assertTrue(True)

    def test_pagexml_to_img_id_mapping(self):
        bf = '/home/nicolas/htr/Monasterium'
        xml2id = MonasteriumDataset(bf, subset='train', extract=False).map_pagexml_to_img_id()
        self.assertTrue( len(xml2id) )

    def test_line_extraction_text_count(self):
        bf = '/home/nicolas/htr/Monasterium'
        tf = 'line_imgs'
        ms = MonasteriumDataset(bf, subset='train', extract=False)
        lst = ms.extract_lines(tf, shape='bbox', text_only=True, limit=5)
        self.assertEqual( len(lst), 5)

    def test_line_extraction_bbox_count(self):
        bf = '/home/nicolas/htr/Monasterium'
        tf = 'line_imgs'
        ms = MonasteriumDataset(bf, subset='train', extract=False)
        lst = ms.extract_lines(tf, shape='bbox', limit=5)
        self.assertEqual( len(lst), 5)

    def test_line_extraction_polygon_count(self):
        bf = '/home/nicolas/htr/Monasterium'
        tf = 'line_imgs'
        ms = MonasteriumDataset(bf, subset='train', extract=False)
        lst = ms.extract_lines(tf, limit=5)
        self.assertEqual( len(lst), 5)

    def test_line_extraction_file_creation(self):
        bf = '/home/nicolas/htr/Monasterium'
        ms = MonasteriumDataset(bf, subset='train', target_folder='line_imgs', extract=False)
        tf = 'line_imgs'
        ms.extract_lines(tf, limit=5)
        gt_cnt, img_cnt = ms.count_line_items(tf)
        self.assertTrue( gt_cnt == 5 and img_cnt == 5)

    def test_dataset_getitem(self):
        bf = '/home/nicolas/htr/Monasterium'
        ms = MonasteriumDataset(bf, subset='train', target_folder='line_imgs', limit=5)
        self.assertTrue( type( ms[0] ) is tuple and type( ms[0][0] ) is Tensor and type( ms[0][1] ) is str )

    def test_dataset_length(self):
        bf = '/home/nicolas/htr/Monasterium'
        ms = MonasteriumDataset(bf, subset='train', target_folder='line_imgs', limit=5)
        self.assertEqual( len( ms ), 5 )

    def Dtest_dataset_full(self):
        bf = '/home/nicolas/htr/Monasterium'
        ms = MonasteriumDataset(bf, subset='train', target_folder='line_imgs')
        self.assertEqual( len( ms ), 3033 )

    def test_dataset_csv_dump(self):
        bf = '/home/nicolas/htr/Monasterium'
        tf = 'line_imgs'
        ms = MonasteriumDataset(bf, subset='train', target_folder=tf, limit=5)
        self.assertTrue( Path( tf, 'monasterium_ds.csv').exists() )

    def test_dataset_csv_load(self):
        bf = '/home/nicolas/htr/Monasterium'
        tf = 'line_imgs'
        test_csv = """line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l2.png Wir Ruprecht der Elter von Gotes gnaden pfallenczgrave by Rine, dez heiligen Romischen riches obrester truchsesze
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l3.png und herczog in Beyern, bekennen und tun chunt offenlichen mit disem briefe allen den, die in sehent oder horent le¬
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l4.png sen, daz uns kuntlichen und wissenlichen ist, und daz selben unser eltern und vordern herczogen zu Beyern an uns
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l5.png und seliger gedechtnusse den hochgeborn herczog Rudolf, unserm bruder, her bracht haben, daz die vesten und merkte
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l6.png der Hohenstein, Hersprugge und Urbach, die von unsers bruders herczog Rudolfs dez vorgenanten tode uf uns verfallen
"""
        ms = MonasteriumDataset(bf, subset='train', target_folder=tf, limit=5)
        
        csv_file = Path( tf, 'monasterium_ds.csv')
        with open(csv_file, 'w') as of:
            of.write(test_csv)
        ms = MonasteriumDataset(bf, subset='train', target_folder=tf, extract=False)
        if csv_file.exists():
            csv_file.unlink()
        self.assertEqual( len(ms), 5 )



if __name__ == "__main__":
    unittest.main() 

