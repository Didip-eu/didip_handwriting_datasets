import unittest
from handwriting_datasets.iam import IAMDataset
import torchvision.transforms as tvtf

class IAMTest( unittest.TestCase ):

    def Dtest_dataset_construction_for_lines_train(self):
        iam_data = IAMDataset('.', subset='train', task='lines', extract=True)
        self.assertEqual(len(iam_data.get_sample_dictionary()), 6161)

    def Dtest_dataset_construction_for_lines_validation1(self):
        iam_data = IAMDataset('.', subset='validation1', task='lines', extract=True)
        self.assertEqual(len(iam_data.get_sample_dictionary()), 900)

    def Dtest_dataset_construction_for_lines_validation2(self):
        iam_data = IAMDataset('.', subset='validation2', task='lines', extract=True)
        self.assertEqual(len(iam_data.get_sample_dictionary()), 940)

    def Dtest_dataset_construction_for_words(self):
        iam_data = IAMDataset('.', subset='train', task='words', extract=True)
        self.assertEqual(len(iam_data.get_sample_dictionary()), 53839)

    def test_dummy_dataset_construction(self):
        iam_data = IAMDataset('.', subset='train', limit=5, extract=False)
        self.assertEqual(len(iam_data.get_sample_dictionary()), 5)

    def test_text_transform_in_dictionary(self):
        iam_data = IAMDataset('.', subset='train', limit=5, extract=False, target_transform=lambda x: f'#{x}#').get_sample_dictionary()
        self.assertTrue( iam_data[0]['text'][0]=='#' and iam_data[0]['text'][-1]=='#' )

    def test_key_access_with_default_transform(self):
        iam_data = IAMDataset('.', subset='validation1', limit=5, extract=False)
        self.assertEqual( len(iam_data[0][0]), 3)

    def test_key_access_with_identity_transform(self):
        iam_data = IAMDataset('.', subset='validation1', limit=5, extract=False, transform=lambda x: x)
        self.assertEqual( len(iam_data[0][0]), 3)

    def test_key_access_with_torchvision_transforms(self):
        transform = tvtf.Compose( [ tvtf.CenterCrop(5)] )
        iam_data = IAMDataset('.', subset='validation1', limit=5, extract=False, transform=transform)
        self.assertEqual( iam_data[0][0][1:], (5,5) )

if __name__ == "__main__":
    unittest.main()

