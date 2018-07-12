import trajnettools


def test_start_enc():
    input_embedding = trajnettools.lstm.modules.InputEmbedding(4, 1.0)
    assert input_embedding.start_enc(2).numpy().tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]


def test_start_dec():
    input_embedding = trajnettools.lstm.modules.InputEmbedding(4, 1.0)
    assert input_embedding.start_dec(2).numpy().tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
