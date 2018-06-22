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


## Average L2 [m]
                               |  Lin | LSTM
                    biwi_eth/* | 0.71 | 0.85
                  biwi_hotel/* | 0.56 | 0.71
               crowds_zara01/* | 0.61 | 0.65
               crowds_zara02/* | 0.67 | 0.65
         crowds_uni_examples/* | 0.73 | 0.81

## Average L2 (non-linear sequences) [m]
                               |  Lin | LSTM
                    biwi_eth/* | 1.19 | 1.00
                  biwi_hotel/* | 0.98 | 0.96
               crowds_zara01/* | 1.13 | 0.86
               crowds_zara02/* | 1.26 | 0.94
         crowds_uni_examples/* | 1.45 | 1.18

## Final L2 [m]
                               |  Lin | LSTM
                    biwi_eth/* | 1.28 | 1.29
                  biwi_hotel/* | 0.97 | 1.09
               crowds_zara01/* | 1.08 | 1.05
               crowds_zara02/* | 1.19 | 1.11
         crowds_uni_examples/* | 1.30 | 1.36
