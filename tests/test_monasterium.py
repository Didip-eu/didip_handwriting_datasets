import unittest
from handwriting_datasets.monasterium import MonasteriumDataset
from torch import Tensor
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

class MonasteriumTest( unittest.TestCase ):

    # 
    bf = '/home/nicolas/tmp/data'
    tf = '/home/nicolas/tmp/data/Monasterium'

    def test_dataset_dummy(self):
        monasterium_data = MonasteriumDataset(self.bf, extract_pages=False)
        self.assertEqual( monasterium_data.data, [])

    def test_download_archive(self):
        monasterium_data = MonasteriumDataset(self.bf, extract_pages=False)
        self.assertEqual( monasterium_data.data, [])

    def test_line_extraction_text_count(self):
        ls = MonasteriumDataset(self.bf, extract_lines=False).extract_lines(self.tf, shape='bbox', text_only=True, limit=5)
        self.assertEqual( len(ls), 5)

    def test_line_extraction_bbox_count(self):
        ls = MonasteriumDataset(self.bf, extract_lines=False).extract_lines(self.tf, shape='bbox', limit=5)
        self.assertEqual( len(ls), 5)

    def Dtest_split_set_train(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        ls = ms.extract_lines(self.tf, shape='bbox', limit=10)
        self.assertEqual( len(ms.split_set( ls, 'train')), 7)

    def Dtest_split_set_validate(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        ls = ms.extract_lines(self.tf, shape='bbox', limit=10)
        self.assertEqual( len(ms.split_set( ls, 'validate')), 1)

    def Dtest_split_set_test(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        ls = ms.extract_lines(self.tf, shape='bbox', limit=10)
        self.assertEqual( len(ms.split_set( ls, 'test')), 2)

    def Dtest_split_set_default(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        ls = ms.extract_lines(self.tf, shape='bbox', limit=10)
        self.assertEqual( len(ms.split_set( ls )), 7)

    def Dtest_split_set_no_overlap(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        ls = ms.extract_lines(self.tf, shape='bbox', limit=10)
        train_set = set( ms.split_set( ls, 'train'))
        validate_set = set( ms.split_set( ls, 'validate'))
        test_set = set( ms.split_set( ls, 'test'))
        self.assertFalse(
                train_set.intersection( validate_set )
             or train_set.intersection( test_set )
             or validate_set.intersection( test_set ))

    def test_line_extraction_polygon_count(self):
        ms = MonasteriumDataset(self.bf, extract_lines=False)
        lst = ms.extract_lines(self.tf, limit=5)
        self.assertEqual( len(lst), 5)

    def test_line_extraction_file_creation(self):
        ms = MonasteriumDataset(self.bf, target_folder='line_imgs', extract_lines=False)
        ms.extract_lines(self.tf, limit=5)
        gt_cnt, img_cnt = ms.count_line_items(self.tf)
        self.assertTrue( gt_cnt == 5 and img_cnt == 5)

    def Dtest_dataset_getitem(self):
        ms = MonasteriumDataset(self.bf, subset='train', target_folder='line_imgs', limit=10)
        self.assertTrue( type( ms[0] ) is tuple and type( ms[0][0] ) is Tensor and type( ms[0][1] ) is str )

    def test_dataset_lines_count(self):
        ms = MonasteriumDataset(self.bf, subset='train', target_folder='line_imgs', extract_lines=True, line_count=10)
        self.assertEqual( len( ms ), 7 )

    def test_dataset_lines_full(self):
        ms = MonasteriumDataset(self.bf, subset='train', target_folder='line_imgs', extract_lines=True)
        self.assertEqual( len( ms ), 5897 )

    def Dtest_dataset_lines_csv_dump(self):
        ms = MonasteriumDataset(self.bf, subset='train', target_folder=self.tf, limit=10)
        self.assertTrue( Path(self.tf, 'monasterium_ds.csv').exists() )

    def Dtest_dataset_lines_csv_load(self):
        test_csv = """line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l2.png Wir Ruprecht der Elter von Gotes gnaden pfallenczgrave by Rine, dez heiligen Romischen riches obrester truchsesze
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l3.png und herczog in Beyern, bekennen und tun chunt offenlichen mit disem briefe allen den, die in sehent oder horent le¬
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l4.png sen, daz uns kuntlichen und wissenlichen ist, und daz selben unser eltern und vordern herczogen zu Beyern an uns
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l5.png und seliger gedechtnusse den hochgeborn herczog Rudolf, unserm bruder, her bracht haben, daz die vesten und merkte
line_imgs/UEATCCLUTCEPCEOJQYTDXTXU-r1l6.png der Hohenstein, Hersprugge und Urbach, die von unsers bruders herczog Rudolfs dez vorgenanten tode uf uns verfallen
"""
        ms = MonasteriumDataset(self.bf, subset='train', target_folder=self.tf, limit=5)
        
        target_folder = 'tests'
        csv_file = Path(target_folder, 'monasterium_ds.csv')
        with open(csv_file, 'w') as of:
            of.write(test_csv)
        ms = MonasteriumDataset(self.bf, subset='train', target_folder=target_folder, extract_pages=False)
        if csv_file.exists():
            csv_file.unlink()
        self.assertEqual( len(ms), 5 )

    def Dtest_gentle_fail_on_empty_transcriptions(self):

        ms_with_empty_gt_lines = "MonasteriumTekliaGTDataset/NA-ACK_13231001_00103_r.xml"

        


if __name__ == "__main__":
    unittest.main() 

