
import os
import pysparkling
import trajnettools


def eval(input_files):
    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(trajnettools.readers.trajnet)
             .cache())
    kalman_predictions = (paths
                          .mapValues(lambda paths: paths[0])
                          .mapValues(trajnettools.kalman.predict))

    paths = (paths
             .mapValues(lambda gts: gts[0])
             .leftOuterJoin(kalman_predictions)
             .cache())

    # for scene, v in paths.mapValues(trajnettools.metrics.average_l2).toLocalIterator():
    #     print(scene, v)
    average_l2 = paths.values().map(trajnettools.metrics.average_l2).mean()
    final_l2 = paths.values().map(trajnettools.metrics.final_l2).mean()
    print(input_files + ' -- average l2:', average_l2)
    print(input_files + ' -- final l2:', final_l2)


if __name__ == '__main__':
    eval('output/test/biwi_eth/*.txt')
    eval('output/train/biwi_hotel/*.txt')
    eval('output/test/crowds_zara01/*.txt')
    eval('output/train/crowds_zara02/*.txt')
    eval('output/test/crowds_uni_examples/*.txt')
    # eval('output/train/crowds_students001/*.txt')
    # eval('output/train/crowds_students003/*.txt')

