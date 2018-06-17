"""Create Trajnet data from original datasets."""

import os
import pysparkling
import trajnettools


def biwi(sc, input_file, output_file):
    biwi_input = (sc
                  .textFile(input_file)
                  .map(trajnettools.readers.biwi)
                  .cache())

    for scene_id, ped_id, rows in trajnettools.scene.to_scenes(biwi_input):
        (rows
         .map(lambda r: trajnettools.writers.trajnet_mark(r, ped_id))
         .saveAsTextFile(output_file.format(scene_id)))


def crowds(sc, input_file, output_file):
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
    mot_input = sc.parallelize(trajnettools.readers.mot(input_file))

    for scene_id, ped_id, rows in trajnettools.scene.to_scenes(mot_input, frame_rate=7.0):
        (rows
         .map(lambda r: trajnettools.writers.trajnet_mark(r, ped_id))
         .saveAsTextFile(output_file.format(scene_id)))


def main():
    sc = pysparkling.Context()
    # biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt', 'data/train/biwi_hotel/{}.txt')
    # crowds(sc, 'data/raw/crowds/arxiepiskopi1.vsp', 'data/train/crowds_arxiepiskopi1/{}.txt')
    # crowds(sc, 'data/raw/crowds/crowds_zara02.vsp', 'data/train/crowds_zara02/{}.txt')
    # crowds(sc, 'data/raw/crowds/crowds_zara03.vsp', 'data/train/crowds_zara03/{}.txt')
    # crowds(sc, 'data/raw/crowds/students001.vsp', 'data/train/crowds_students001/{}.txt')
    # crowds(sc, 'data/raw/crowds/students003.vsp', 'data/train/crowds_students003/{}.txt')
    mot(sc, 'data/raw/mot/PETS2009-S2L1.xml', 'data/train/mot_pets2009_s2l1/{}.txt')

if __name__ == '__main__':
    main()
