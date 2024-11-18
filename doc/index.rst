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


How to use the ChartersDataset class
====================================

----------------------------------------------------------------------
Working from an existing dataset archive → subclassing ChartersDataset
----------------------------------------------------------------------

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


Alphabets: notes for myself
===========================

-----------------
Desired features
-----------------


* An alphabet is a glorified dictionary; if it is many-to-one (*n* symbols map to 1 code), the operation
  that maps a code to a symbol (for decoding) is consistently returns the same symbol, no matter how the
  dictionary was created.
* No matter how it has been created (from a in-memory dictionary or a file), the alphabet used for instantiating
  a dataset is ultimately stored in a file ``alphabet.tsv`` in the same directory as the work samples.
* Any dataset that is read from an existing TSV file is assigned by default the alphabet stored in ``alphabet.tsv``.
* The ``alphabet`` attribute of a dataset can be assigned with an Alphabet object, that overrides the default:
  + at initialization time, this alphabet overrides the on-disk alphabet (``alphabet.tsv`` file).
  + if the alphabet is set in a later stage, the on-disk alphabet is not modified.
  This puts a premimum on the alphabet used for creating the dataset on-disk; it can easily be read back and
  examined later (and modified on-disk, if ever needed). Assigning an alphabet on a live dataset object at a
  latere stage may be useful, but it should not mess with the existing on-disk data.

The ChartersDataset class initializes a default alphabet, that can be replaced
>>>>>>> 9ce1312 (Doc: alphabet subsection.)

-----------------------------------------
Fitting data actual charset to a model
-----------------------------------------

Distinguish between:

1. symbols we want to ignore/discard, because they're junk or
   irrelevant for any HTR purpose - they typically do not have
   any counterpart in the input image.

2. symbols that may have a counterpart in the original image,
   but we want to ignore now because they don't matter for our
   particular experiment - by they may do for the next one.
   It is the HTR analog of '> /dev/null' !

3. symbols we don't want to interpret but need to be there in some way
   in the output (if just for being able to say: there is something
   here that needs interpreting. Eg. abbreviations)

For (1), they can be filtered out of the sample at any stage
between the page/xlm collation phase and the last, encoding stage,
but it makes sense to do it as early as possible, to avoid any
overhead later, while not having to think too much about it (i.e.
avoid having to use a zoo of pre-processing scripts that live
outside this module): the line extraction method seems a good place
to do that, and no alphabet is normally needed.

For (2) and (3), we want more flexibility: what is to be ignored or
not interpreted now may change tomorrow. The costly line extraction
routine, that is meant to be run during the early dataset setup and
generally once, is not the best place to do it: rather, the
getitem transforms are appropriate. But how?

* to ignore a symbol, just do not include it into the alphabet: it
  will map to null at the encoding stage, while still occupy one slot
  in the sequence, which is what we want;

* a symbol that is not to be interpreted needs to be in the alphabet,
  even though it is just a normal symbol, that maps to a conventional code
  for 'unknown'. Eg. a character class '?' may contain all tricky baseline
  abbreviations -> it must be easy to merge different classes into
  a single 'unknown' set.

In practice, any experiment needs to define its alphabet, as follows:

1. Define a reasonable, all-purpose default alphabet in the ChartersDataset
   class, that leaves most options open, while still excluding the most
   unlikely ones. Eg.  an alphabet without the Hebrew characters, diacritic marks, etc.
   It should be a class property, so that any addition to the character classes
   is duly reflected in the default.

2. Construct a specific alphabet out of it, by substraction or merging.
   A that point, the class 'unknown' is just like another class: all
   subsets of characters that make it should be merged accordingly first.
   Only then does the finalization method define a stand-in symbol

for this class, through a quick lookup at any character that is member
of the set (see `unknown_class_representant` option in the initialization
function).




.. toctree::
   :maxdepth: 3
   :caption: Contents:


ChartersDataset
================

.. automodule:: didip_handwriting_datasets.charters
   :members:
   :member-order: groupwise


Alphabet
===========



.. automodule:: didip_handwriting_datasets.alphabet
   :members:


