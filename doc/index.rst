.. DiDip Handwriting Datasets documentation master file, created by
   sphinx-quickstart on Sun Nov  3 09:57:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************
DiDip Handwriting Datasets
***************************

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


How to use
=============

------------------------------------------------------
Working from an existing dataset archive → subclassing
------------------------------------------------------

This is probably the most common use of the `ChartersDataset` class: assuming that the dataset 
is provided as an archive of page images and their metadata, that we want to keep 
separate from other sources, we create a subclass with the appropriate archive metadata
(the `dataset_resource` class attribute), as illustrated by the `MonasteriumDataset` and
`KoenigsfeldenDataset` classes below in this module. Setting a couple of class attributes
allows for specific default root and work directories to be created, with no risk of overriding
an existing folder.

As an example, the following line decompresses all images and pageXML files contained in the Koenigsfelden
archive into a Koenigsfelden root folder:

.. code-block::

>>> monasterium.KoenigsfeldenDataset(extract_pages=True)

(Since the archive extraction stage may takes a few minutes, the default is to assume that it has already
been decompressed.)


----------------------------------------------------------------------------
Create a dataset from arbitrary pages or line items -> direct instantiation  
----------------------------------------------------------------------------

Assuming that a directory already contains some data (whose provenance does not matter), we want to
create a dataset object that is fit for training or inference tasks:

* if the directory contains page images ``%.jpg`` and their corresponding PageXML metadata ``%.xml``,
  the ``-from_page_xml_dir`` option allows for building the line images and transcriptions into a
  task-specific work folder, as well as a corresponding TSV file mapping images to their sizes
  and transcriptions.

* if a folder already contains line images ``%.png`` and transcriptions ``%.gt.txt``, any TSV file
  that lists all or part of those files may be used to create a working dataset:

* if the TSV file does not exist, create it from existing line items as shown here:

  .. code-block::

      >>> trainDS = mom.ChartersDataset(task='htr', build_items=False, subset='train', subset_ratios=(.85,.05,.1))

  The ``build_items=False`` option ensures that the (costly) line extraction routine does not run.
  The provided subset ratios allow to define which proportion of the existing files
  should go into respectively the 'train', 'validate', and 'test' sets. (All 3
  sets are distinct, no matter what.)

* in order to reuse an existing TSV dataset definition, use the ``-from_line_tsv_file`` option:
  it builds a sample set out of the files listed in the provided TSV file---it
  may reference any subset of the files contained in the parent directory, which
  implicitly becomes the work folder for the task.  A single work folder can be used to
  construct different sample sets: just provide different TSV files for each. A typical use
  is to create train, validation, and test TSV specs, out of a single set of physical image/GT files.

  .. code-block::

      >>> trainDS = mom.ChartersDataset(task='htr', from_line_tsv_file='myCustomSet.tsv')


===============================================
Directories: root, raw data, and work folders
===============================================

The workflow assumes the following directory structure: 

* ``root``: where datasets archives are to be downloaded, decompressed and where work folders are to
  be created. Default: ``data/<root_folder_basename>`` where ``data`` is a subfolder in this package
  directory and ``<root_folder_basename>`` is a class attribute. Eg. ``data/Charters``.

* ``root/<raw data folder>`` : where the content of the original archive is to be extracted, i.e. a
  subdirectory containing all image and GT files. Eg. ``data/Charters/ChartersRawData``.

* ``work_folder``: where to create the dataset to be used for given task (segmentation or htr).
  Default: ``<root>/<work_folder_name class attribute><task suffix>``, eg. ``data/Charters/ChartersHandwritingDatasetHTR``

The first two directories can be created with a 'no-task' (default) initialization and forcing the archive extraction:

.. code-block:: python

>>> ChartersDataset( extract_pages=True )

The following call leaves the first two folders untouched and uses their data to build a task-specific instance:

.. code-block:: python

>>> myHtrDs = ChartersDataset( task='htr' )



.. toctree::
   :maxdepth: 3
   :caption: Contents:

.. automodule:: didip_handwriting_datasets.monasterium
   :members:
   :member-order: groupwise

.. automodule:: didip_handwriting_datasets.alphabet
   :members:


