
from contextlib import contextmanager
import matplotlib.pyplot as plt
import pysparkling
import trajnettools


@contextmanager
def show(fig_file=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=300)
    fig.show()
    plt.close(fig)


def predict(input_files):
    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(trajnettools.readers.trajnet)
             .cache())
    kalman_predictions = (paths
                          .mapValues(lambda paths: paths[0])
                          .mapValues(trajnettools.kalman.predict))

    paths = paths.leftOuterJoin(kalman_predictions)
    for i, (scene, (gt, kf)) in enumerate(paths.collect()):
        with show('output/biwi_hotel_scene{}.png'.format(i)) as ax:
            # KF prediction
            ax.plot([gt[0][8].x] + [r.x for r in kf],
                    [gt[0][8].y] + [r.y for r in kf], color='orange', label='KF')
            ax.plot([kf[-1].x], [kf[-1].y], color='orange', marker='x', linestyle='None')

            # ground truths
            for i_gt, g in enumerate(gt):
                xs = [r.x for r in g]
                ys = [r.y for r in g]

                # markers
                label_start = None
                label_end = None
                if i_gt == 0:
                    label_start = 'start'
                    label_end = 'end'
                ax.plot(xs[0:1], ys[0:1], color='black', marker='o', label=label_start, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='x', label=label_end, linestyle='None')

                # ground truth lines
                ls = 'dotted' if i_gt > 0 else 'solid'
                label = None
                if i_gt == 0:
                    label = 'primary'
                if i_gt == 1:
                    label = 'others'
                ax.plot(xs, ys, color='black', linestyle=ls, label=label)

            # frame
            ax.legend()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')


if __name__ == '__main__':
    predict('data/train/biwi_hotel/?.txt')
