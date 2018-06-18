## Prepare Data

Existing data:

.. code-block::

    data/
        data_arxiepiskopi.rar
        data_university_students.rar
        data_zara.rar
        ewap_dataset_light.tgz
        3DMOT2015Labels  # from: https://motchallenge.net/data/3DMOT2015Labels.zip (video file at http://cs.binghamton.edu/~mrldata/public/PETS2009/S2_L1.tar.bz2)
        Train.zip  # from trajnet.epfl.ch

Extract:

.. code-block:: sh

    # biwi
    mkdir -p data/raw/biwi
    tar -xzf data/ewap_dataset_light.tgz --strip-components=1 -C data/raw/biwi

    # crowds
    mkdir -p data/raw/crowds
    unrar e data/data_arxiepiskopi.rar data/raw/crowds
    unrar e data/data_university_students.rar data/raw/crowds
    unrar e data/data_zara.rar data/raw/crowds

    # PETS09 S2L1 ground truth
    mkdir -p data/raw/mot
    cp data/3DMOT2015Labels/train/PETS09-S2L1/gt/gt.txt data/raw/mot/pets2009_s2l1.txt

    # original Trajnet files
    mkdir -p data/trajnet_original
    tar -xzf data/Train.zip -C data/trajnet_original
    mv data/trajnet_original/train/* data/trajnet_original
    rm -r data/trajnet_original/train
    rm -r data/trajnet_original/__MACOSX


## Difference in generated data

* partial tracks are now included (for correct occupancy maps)
* pedestrians that appear in multiple chunks had the same id before (might be a problem for some input readers)
* separate scenes with annotation of the one primary pedestrian
* the primary pedestrian has to move by more than 1 meter
