
~~~~
cd didip_handwriting_datasets
pip install .
~~~~~~~~~~

The lifecycle of a dataset instance relies on 3 folders:

|    | Step              | Folder option           |                                                   | Configurable                                                               |
| -- | ----------------- | ----------------------- | ------------------------------------------------- |----------------------------------------------------------------------------|
| 1  | Downloading       | `<root>`                | location where original archives are to be stored | Y (`root=`)                                                                |
| 2  | Extract           | `<root>/<base_folder>`  | where archive's contents are extracted            | N (name hardcoded for dataset Eg. `'MonasteriumHandwritingDataset'`)       |
| 3  | Instance creation | `<work_folder>`         | where input data for the task are created         | Y (default: <code>'./MonasteriumHandwritingDataset(HTR\|Segment)'</code>)  |

1. By default, the root folder (default: `~/tmp/data/Monasterium)` is checked first for a valid archive before downloading;
2. The base folder is automatically created when expanding the archive (default: `<root>/MonasteriumHandwritingDataset`); this location is checked for a valid file tree (checksums). To force the extraction of the archive's content, pass the `extract_pages=True` flag. 
3. For most use cases, it is sufficient to specify the work folder (option `work_folder`), where the ready-to-use data (line images and ground truth files) for the task at hand are to be created. By default, a folder named `MonasteriumHandwritingDatasetHTR` is created in the current work directory.

Examples for HTR:

~~~python

from didip_handwriting_datasets import monasterium

# create new dataset instance for HTR (line image and ground-truth files) in ~/tmp/data/MyHTRExperiment
# The root folder is assumed to be the default location (/tmp/data/Monasterium/)
myDataSet = monasterium.MonasteriumDataset(work_folder='~/tmp/data/MyHTRExperiment')

# the following assumes that the dataset archive file is to be downloaded (if needed) and read from '/home/nicolas/tmp/didip'
myDataSet = monasterium.MonasteriumDataset(root='/home/nicolas/tmp/didip', work_folder='~/tmp/data/MyHTRExperiment')

# create a BBox segmentation dataset in the default work folder ('./MonasteriumHandwritingDatasetSegment')
myDataSet = monasterium.MonasteriumDataset(task='segment', shape='bbox')
~~~~~~~~



