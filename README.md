## Install

~~~~
cd didip_handwriting_datasets
pip install .
~~~~~~~~~~

## Dataset classes, instances and folder

The lifecycle of a dataset instance relies on 3 stages. With the Monasterium HW dataset as an example:

|    | Step              | Folder option    |                                                   | Configurable                                                     |
| -- | ----------------- | ---------------- | ------------------------------------------------- |------------------------------------------------------------------|
| 1  | Downloading       | `<root>`         | location where original archives are to be stored | Y (`root=<path>`) - Default: `./<project_dir>/data/Monasterium`  |
| 2  | Extract           |                  | where archive's contents are extracted            | N (`<root>/MonasteriumTekliaGTDataset`)                          |
| 3  | Instance creation | `<work_folder>`  | where input data for the task are created         | Y (default: `<root>/MonasteriumHandwritingDataset(HTR\|Segment)` |

1. By default, the root folder is checked first for a valid archive before downloading;
2. The base folder is automatically created when expanding the archive (default: `<root>/MonasteriumHandwritingDataset`); this location is checked for a valid file tree (checksums). To force the extraction of the archive's content, pass the `extract_pages=True` flag. 
3. The work folder is the place where the ready-to-use data (line images and ground truth files) for the task at hand are to be created

    + if no work folder is specified, a folder `MonasteriumHandwritingDatasetHTR` (for an HTR task) or `MonasteriumHandwritingDatasetSegment` (for a segmentation task) will be created under the `root` folder.
    + if the option `work_folder`) is passed an absolute path, such a folder will be created (including missing parents)
    + if the option `work_folder`) is passed a relative path, a corresponding folder (including missing parents) is created under the `root` folder.
 

## How to use

### Working from an existing dataset archive → subclassing

This is probably the most common use of the `ChartersDataset` class: assuming that the dataset 
is provided as an archive of page images and their metadata, that we want to keep 
separate from other sources, we create a subclass with the appropriate archive metadata
(the `dataset_resource` class attribute), as illustrated by the `MonasteriumDataset` and
`KoenigsfeldenDataset` classes below in this module. Setting a couple of class attributes
allows for specific default root and work directories to be created, with no risk of overriding
an existing folder.

As an example, the following line decompresses all images and pageXML files contained in the Koenigsfelden
archive into a Koenigsfelden root folder:


```python
monasterium.KoenigsfeldenDataset(extract_pages=True)
```

(Since the archive extraction stage may takes a few minutes, the default is to assume that it has already
been decompressed.)


### Create a dataset from arbitrary pages or line items -> direct instantiation  

Assuming that a directory already contains some data (whose provenance does not matter), we want to
create a dataset object that is fit for training or inference tasks:

* if the directory contains page images `%.jpg` and their corresponding PageXML metadata `%.xml`,
  the `-from_page_xml_dir` option allows for building the line images and transcriptions into a
  task-specific work folder, as well as a corresponding TSV file mapping images to their sizes
  and transcriptions.

* if a folder already contains line images `%.png` and transcriptions `%.gt.txt`, any TSV file
  that lists all or part of those files may be used to create a working dataset:

* if the TSV file does not exist, create it from existing line items as shown here:

```python
>>> trainDS = mom.ChartersDataset(task='htr', build_items=False, subset='train', subset_ratios=(.85,.05,.1))
```

  The `build_items=False` option ensures that the (costly) line extraction routine does not run.
  The provided subset ratios allow to define which proportion of the existing files
  should go into respectively the 'train', 'validate', and 'test' sets. (All 3
  sets are distinct, no matter what.)

* in order to reuse an existing TSV dataset definition, use the `-from_line_tsv_file` option:
  it builds a sample set out of the files listed in the provided TSV file---it
  may reference any subset of the files contained in the parent directory, which
  implicitly becomes the work folder for the task.  A single work folder can be used to
  construct different sample sets: just provide different TSV files for each. A typical use
  is to create train, validation, and test TSV specs, out of a single set of physical image/GT files.


```python
>>> trainDS = mom.ChartersDataset(task='htr', from_line_tsv_file='myCustomSet.tsv')
```

### Directories: root, raw data, and work folders


The workflow assumes the following directory structure: 

* `root`: where datasets archives are to be downloaded, decompressed and where work folders are to
  be created. Default: `data/<root_folder_basename>` where `data` is a subfolder in this package
  directory and `<root_folder_basename>` is a class attribute. Eg. `data/Charters`.

* `root/<raw data folder>` : where the content of the original archive is to be extracted, i.e. a
  subdirectory containing all image and GT files. Eg. `data/Charters/ChartersRawData`.

* `work_folder`: where to create the dataset to be used for given task (segmentation or htr).
  Default: `<root>/<work_folder_name class attribute><task suffix>`, eg. `data/Charters/ChartersHandwritingDatasetHTR`

The first two directories can be created with a 'no-task' (default) initialization and forcing the archive extraction:

```python
>>> ChartersDataset( extract_pages=True )
```

The following call leaves the first two folders untouched and uses their data to build a task-specific instance:


```python
>>> myHtrDs = ChartersDataset( task='htr' )
```




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




