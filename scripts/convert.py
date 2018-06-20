"""Create Trajnet data from original datasets."""

import os
import pysparkling
import trajnettools


def biwi(sc, input_file, output_file):
    print('processing ' + input_file)
    biwi_input = (sc
                  .textFile(input_file)
                  .map(trajnettools.readers.biwi)
                  .cache())

    for scene_id, ped_id, rows in trajnettools.scene.to_scenes(biwi_input):
        (rows
         .map(lambda r: trajnettools.writers.trajnet_mark(r, ped_id))
         .saveAsTextFile(output_file.format(scene_id)))


def crowds(sc, input_file, output_file):
    print('processing ' + input_file)
    crowds_input = (sc
                    .wholeTextFiles(input_file)
                    .values()
                    .flatMap(trajnettools.readers.crowds)
                    .cache())

    for scene_id, ped_id, rows in trajnettools.scene.to_scenes(crowds_input):
        (rows
         .map(lambda r: trajnettools.writers.trajnet_mark(r, ped_id))
         .saveAsTextFile(output_file.format(scene_id)))


def mot(sc, input_file, output_file):
    """Supposedly was 7 frames per second in original recording."""
    print('processing ' + input_file)
    mot_input = (sc
                 .textFile(input_file)
                 .map(trajnettools.readers.mot)
                 .filter(lambda r: r.frame % 2 == 0)
                 .cache())

    for scene_id, ped_id, rows in trajnettools.scene.to_scenes(mot_input):
        (rows
         .map(lambda r: trajnettools.writers.trajnet_mark(r, ped_id))
         .saveAsTextFile(output_file.format(scene_id)))


def main():
    sc = pysparkling.Context()

    # train
    biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt', 'output/train/biwi_hotel/{}.txt')
    crowds(sc, 'data/raw/crowds/arxiepiskopi1.vsp', 'output/train/crowds_arxiepiskopi1/{}.txt')
    crowds(sc, 'data/raw/crowds/crowds_zara02.vsp', 'output/train/crowds_zara02/{}.txt')
    crowds(sc, 'data/raw/crowds/crowds_zara03.vsp', 'output/train/crowds_zara03/{}.txt')
    crowds(sc, 'data/raw/crowds/students001.vsp', 'output/train/crowds_students001/{}.txt')
    crowds(sc, 'data/raw/crowds/students003.vsp', 'output/train/crowds_students003/{}.txt')
    mot(sc, 'data/raw/mot/pets2009_s2l1.txt', 'output/train/mot_pets2009_s2l1/{}.txt')

    # test
    biwi(sc, 'data/raw/biwi/seq_eth/obsmat.txt', 'output/test/biwi_eth/{}.txt')
    crowds(sc, 'data/raw/crowds/crowds_zara01.vsp', 'output/test/crowds_zara01/{}.txt')
    crowds(sc, 'data/raw/crowds/uni_examples.vsp', 'output/test/crowds_uni_examples/{}.txt')


if __name__ == '__main__':
    main()
