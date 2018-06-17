def trajnet(row):
    return '{r.frame:d} {r.pedestrian:d} {r.x:.2f} {r.y:.2f}'.format(r=row)


def trajnet_mark(row, marked_id):
    return ('{r.frame:d} {r.pedestrian:d} {r.x:.2f} {r.y:.2f} {m:d}'
            ''.format(r=row, m=(marked_id == row.pedestrian)))
