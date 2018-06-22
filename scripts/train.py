import pysparkling
import trajnettools


def train(input_files):
    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(trajnettools.readers.trajnet)
             .cache())

    # LSTM training
    training_paths = paths.values().map(lambda paths: paths[0]).collect()
    lstm_predictor = trajnettools.sociallstm.train(training_paths)
    lstm_predictor.save('output/lstm.pkl')


if __name__ == '__main__':
    train('output/test/biwi_eth/*.txt')
