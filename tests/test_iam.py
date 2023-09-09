import unittest
from handwriting_datasets.iam import IAMDataset

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

if __name__ == "__main__":
    unittest.main()

