## Install

~~~~
cd didip_handwriting_datasets
pip install .
~~~~~~~~~~

## Handling datasets

### Lifecycle view: the states of a dataset

The lifecycle of a dataset instance relies on 3 stages, withe MonasteriumTeklia dataset as an example:

|    | Step              | Folder option    |                                                   | Configurable                                                     |
| -- | ----------------- | ---------------- | ------------------------------------------------- |------------------------------------------------------------------|
| 1  | Downloading       | `<root>`         | location where original archives are to be stored | Y (`root=<path>`) - Default: `./<project_dir>/data/Charters`  |
| 2  | Extract           |                  | where archive's contents are extracted            | N (`<root>/MonasteriumTekliaGTDataset`)                          |
| 3  | Instance creation | `<work_folder>`  | where input data for the task are created         | Y (default: `./data/MonasteriumHandwritingDatasetHTR` |

1. By default, the root folder is checked first for a valid archive before downloading;
2. The base folder is automatically created when expanding the archive (default: `<root>/MonasteriumHandwritingDataset`); this location is checked for a valid file tree (checksums). To force the extraction of the archive's content, pass the `extract_pages=True` flag. 
3. The work folder is the place where the ready-to-use data (line images and ground truth files) for the task at hand are to be created

   + if no work folder is specified, a folder `data/MonasteriumHandwritingDatasetHTR` is created in the **current directory**.
   + if the option `work_folder` is set, such a folder will be created, including missing parents. 
 
### Folder view: where are the data?

The workflow assumes the following directory structure: 

* `root`: where datasets archives are to be downloaded, decompressed and where work folders are to
  be created. Default: `data/<root_folder_basename>` where `data` is a subfolder in this package
  directory and `<root_folder_basename>` is a class attribute. Eg. `data/Charters`.

* `root/<raw data folder>` : where the content of the original archive is to be extracted, i.e. a
  subdirectory containing all image and GT files. Eg. `data/Charters/ChartersRawData`.

* `work_folder`: where to create the dataset to be used for given task 
  Default: `./data/<work_folder_name class attribute>`, eg. `data/Charters/ChartersHandwritingDatasetHTR`


## How to use

Depending on the dataset stage, a few use cases, from the easiest to the more complex:

### From a TSV file 

Assumption: the TSV file is in the same location as the pre-compiled line images and transcription files, which is thus the work folder.

```bash
myDataSet=charters_htr.ChartersDataset(from_line_tsv_file='./data/fsdb_aligned_matrix/charter_ds_train.tsv')
```

This command does not create the work folder nor changes the sample files.

### From a work folder of pre-compiled line images and transcription files, generate TSV files

Just generate 3 TSV files for the training, validation, and test tests, respectively. Ex.

```bash
cd ~/vre/ddpa_htr/data/fsdb_aligned_matrix
myDataSet=charters_htr.ChartersDataset(from_work_folder='.', subset_ratios=(.8,.05,.15))
```

By default, the dataset returned by the constructor is the training dataset. Use the `subset=<set>` keyword parameter to choose another subset.
This command does not create the work folder nor changes the sample files.


### From a directory of page images and metadata files 

The following command compiles an existing set of page images and their XML metadata (the default) into a work folder of line items (images and transcriptions):

```bash
charters_htr.ChartersDataset(from_page_dir=f'{os.environ["HOME"]}/tmp/data/Monasterium/MonasteriumTekliaGTDataset', work_folder='./data/MonasteriumTeklia')
```

For non-PageXML metadata, just pass the `gt_suffix` keyword parameter:

```bash
charters_htr.ChartersDataset(from_page_dir=f'{os.environ["HOME"]}/tmp/data/fsdb_work/fsdb_full_text_sample_1000/htr_gt', work_folder='./data/fsdb_aligned_matrix', suffix='htr.gt .json')
```

In both cases, the work folder is created if it does not exist; if it exists, its content is deleted before the compilation.

### Working from an existing dataset archive → subclassing

Assuming that the dataset 
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










