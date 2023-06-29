import unittest
from handwriting_datasets.funsd import FunsdDataset

# Word dataset

class FunsdTest( unittest.TestCase ):

    def test_dataset_construction_for_lines_train(self): 
        funsd_data = FunsdDataset('.', subset='train', extract=True)
        self.assertEqual(len(funsd_data.get_sample_dictionary()), 21888)

    def test_dataset_construction_for_lines_validation(self): 
        funsd_data = FunsdDataset('.', subset='validate', extract=True)
        self.assertEqual(len(funsd_data.get_sample_dictionary()), 4208) 
    
    def test_dataset_construction_for_lines_test(self): 
        funsd_data = FunsdDataset('.', subset='test', extract=True)
        self.assertEqual(len(funsd_data.get_sample_dictionary()), 4499)
    
        
if __name__ == "__main__":
    unittest.main()

