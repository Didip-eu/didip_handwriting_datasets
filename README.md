## Install

~~~~
cd didip_handwriting_datasets
pip install .
~~~~~~~~~~

## Creating a dataset instances: configuration



The lifecycle of a dataset instance relies on 3 stages. With the Monasterium HW dataset as an example:

|    | Step              | Folder option           |                                                   | Configurable                                                               |
| -- | ----------------- | ----------------------- | ------------------------------------------------- |----------------------------------------------------------------------------|
| 1  | Downloading       | `<root>`                | location where original archives are to be stored | Y (`root=<path>`) Default: `./didip_handwriting_datasets/data/Monasterium` |
| 2  | Extract           | `<root>/MonasteriumHandwritingDataset` | where archive's contents are extracted            | N        |
| 3  | Instance creation | `<work_folder>`         | where input data for the task are created         | Y (default: `<root>/MonasteriumHandwritingDataset(HTR\|Segment)`  |

1. By default, the root folder is checked first for a valid archive before downloading;
2. The base folder is automatically created when expanding the archive (default: `<root>/MonasteriumHandwritingDataset`); this location is checked for a valid file tree (checksums). To force the extraction of the archive's content, pass the `extract_pages=True` flag. 
3. The work folder is the place where the ready-to-use data (line images and ground truth files) for the task at hand are to be created

    + if no work folder is specified, a folder `MonasteriumHandwritingDatasetHTR` (for an HTR task) or `MonasteriumHandwritingDatasetSegment` (for a segmentation task) will be created under the `root` folder.
    + if the option `work_folder`) is passed an absolute path, such a folder will be created (including missing parents)
    + if the option `work_folder`) is passed a relative path, a corresponding folder (including missing parents) is created under the `root` folder.
 

## Use cases 

Default root and work folders, no task defined (data points are not compiled):


    >>> ds=monasterium.MonasteriumDatase()
    >>> print(ds)
    Root folder:	/home/nicolas/graz/htr/didip_handwriting_datasets/didip_handwriting_datasets/data/Monasterium
    Files extracted in:	/home/nicolas/graz/htr/didip_handwriting_datasets/didip_handwriting_datasets/data/Monasterium/MonasteriumTekliaGTDataset
    Task: None defined.
    Work folder:	None
    Data points:	0


Default root and work folders, HTR task: data points are `(<line img>, <transcription text>)` pairs; optional `count` option limits the number of pairs:

    >>> ds=monasterium.MonasteriumDataset(task='htr',count=100)
    >>> print(ds)
    Root folder:	/home/nicolas/graz/htr/didip_handwriting_datasets/didip_handwriting_datasets/data/Monasterium
    Files extracted in:	/home/nicolas/graz/htr/didip_handwriting_datasets/didip_handwriting_datasets/data/Monasterium/MonasteriumTekliaGTDataset
    Task: HTR
    Work folder:	/home/nicolas/graz/htr/didip_handwriting_datasets/didip_handwriting_datasets/data/Monasterium/MonasteriumHandwritingDatasetHTR
    Data points:	100




