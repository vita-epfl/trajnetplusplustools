from collections import defaultdict

from .data import Row


def trajnet(whole_file):
    marked = defaultdict(list)
    others = defaultdict(list)
    for line in whole_file.split('\n'):
        if not line:
            continue
        line = [e for e in line.split(' ') if e != '']
        mark = bool(float(line[4]))
        ped_id = int(float(line[1]))
        row = Row(int(float(line[0])), ped_id, float(line[2]), float(line[3]))
        if mark:
            marked[ped_id].append(row)
        else:
            others[ped_id].append(row)

    return list(marked.values()) + list(others.values())
