The lifecycle of a dataset instance relies on 3 folders:

| | Step | Folder | | Configurable |
| -- | ---  | ---- | -- |
| 1 | Downloading | `<root>` | location where original archives are stored | Y (`root=`)|
| 2 | Extract | `<root>/<base_folder>`  | where archive's contents are extracted | N (name hardcoded for dataset Eg. 'MonasteriumHandwritingDataset') |
| 3 | Instance creation | <

By default, the root folder (default: `~/tmp/data/Monasterium)` is checked first for an existing for a valid archive before downloading; then the base folder (default: `~/tmp/data/Monasterium/MonasteriumHandwritingDataset`) is checked for a valid file tree (checksums). To force the extraction of the archive's content, pass the `extract_pages=True` flag. Therefore, for most use cases, it is sufficient to specify the work folder, where the instance of the data (line images and ground truth files) for this particular task is to be created. Eg.
 
~~~python

from handwriting_datasets import monasterium

# create new dataset instance for HTR (line image and ground-truth files) in ~/tmp/data/MyHTRExperiment
myDataSet = monasterium.MonasteriumDataset(work_folder='~/tmp/data/MyHTRExperiment')

# create a BBox segmentation dataset in the default work folder ('./MonasteriumHandwritingDatasetSegment')
myDataSet = monasterium.MonasteriumDataset(task='segment', shape='bbox')
~~~~~~~~

Result: 

~~~~~~~python

~~~~~~~~~~


