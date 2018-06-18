import numpy as np
import scipy.interpolate
import xml.etree.ElementTree

from .data import MarkedRow, Row


def biwi(line):
    line = [e for e in line.split(' ') if e != '']
    return Row(int(float(line[0]) - 1),  # shift from 1-index to 0-index
               int(float(line[1])),
               float(line[2]),
               float(line[4]))


def crowds_interpolate_person(ped_id, person_xyf):
    xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12
    ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12
    fs = np.array([f for _, _, f in person_xyf])

    kind = 'linear'
    if len(fs) > 5:
        kind = 'cubic'

    x_fn = scipy.interpolate.interp1d(fs, xs, kind=kind)
    y_fn = scipy.interpolate.interp1d(fs, ys, kind=kind)

    frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)
    print(frames)

    return [Row(int(f), ped_id, x, y)
            for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T]


def crowds(whole_file):
    pedestrians = []
    current_pedestrian = []
    for line in whole_file.split('\n'):
        if '- Num of control points' in line or \
           '- the number of splines' in line:
            if current_pedestrian:
                pedestrians.append(current_pedestrian)
            current_pedestrian = []
            continue

        # strip comments
        if ' - ' in line:
            line = line[:line.find(' - ')]

        # tokenize
        entries = [e for e in line.split(' ') if e]
        if len(entries) != 4:
            continue

        x, y, f, _ = entries
        current_pedestrian.append([float(x), float(y), int(f)])

    if current_pedestrian:
        pedestrians.append(current_pedestrian)

    return [row
            for i, p in enumerate(pedestrians)
            for row in crowds_interpolate_person(i, p)]


def mot_xml(file_name):
    """PETS2009 dataset.

    Original frame rate is 7 frames / sec.
    """
    tree = xml.etree.ElementTree.parse(file_name)
    root = tree.getroot()
    for frame in root:
        f = int(frame.attrib['number'])
        if f % 2 != 0:  # reduce to 3.5 rows / sec
            continue

        for ped in frame.find('objectlist'):
            p = ped.attrib['id']
            box = ped.find('box')
            x = box.attrib['xc']
            y = box.attrib['yc']

            yield Row(f, int(p), float(x) / 100.0, float(y) / 100.0)


def mot(line):
    """Line reader for MOT files.

    MOT format:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    line = [e for e in line.split(',') if e != '']
    return Row(int(float(line[0])),
               int(float(line[1])),
               float(line[7]),
               float(line[8]))


def trajnet(line):
    line = [e for e in line.split(' ') if e != '']
    return Row(int(float(line[0])),
               int(float(line[1])),
               float(line[2]),
               float(line[3]))


def trajnet_marked(whole_file):
    for line in whole_file.split('\n'):
        if not line:
            continue
        line = [e for e in line.split(' ') if e != '']
        yield MarkedRow(int(float(line[0])),
                        int(float(line[1])),
                        float(line[2]),
                        float(line[3]),
                        bool(float(line[4])))
