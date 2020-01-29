from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

@contextmanager
def canvas(image_file=None, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    fig.show()
    plt.close(fig)


@contextmanager
def paths(input_paths, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # primary
        xs = [r.x for r in input_paths[0]]
        ys = [r.y for r in input_paths[0]]
        # track
        ax.plot(xs, ys, color='black', linestyle='solid', label='primary',
                marker='o', markersize=2.5, zorder=1.9)
        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                linestyle='None', zorder=0.9)
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                linestyle='None', zorder=0.9)

        # other tracks
        for ped_rows in input_paths[1:]:
            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
            # track
            ax.plot(xs, ys, color='black', linestyle='dotted')

        # frame
        ax.legend()

@contextmanager
def interaction_path(path, neigh, kalman=None, output_file=None, obs_len=9):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # Center
        center = path[obs_len, :]
        path = path - center
        neigh = neigh - center

        # Primary Track
        ax.scatter(path[:, 0], path[:, 1], s=2.5, color='b', label='primary')
        ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label='start point')
        ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label='end point')

        for j in range(neigh.shape[1]):
            ax.plot(neigh[:, j, 0], neigh[:, j, 1], color='g')
            ax.plot(neigh[0, j, 0], neigh[0, j, 1], color='g', marker='o')
            ax.plot(neigh[-1, j, 0], neigh[-1, j, 1], color='r', marker='x')

        # kalman if present
        if kalman is not None:
            kalman = kalman - center
            ax.plot(kalman[:, 0, 0], kalman[:, 0, 1], color='r', label='kalman')

        # frame
        ax.legend()

@contextmanager
def predicted_paths(input_paths, pred_paths, pred_neigh_paths=None, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # primary
        xs = [r.x for r in input_paths[0]]
        ys = [r.y for r in input_paths[0]]
        # track
        ax.plot(xs, ys, color='black', linestyle='solid', label='primary',
                marker='o', markersize=2.5, zorder=1.9)
        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                linestyle='None', zorder=0.9)
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                linestyle='None', zorder=0.9)

        # neigh tracks
        for ped_rows in input_paths[1:]:
            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
            # track
            ax.plot(xs, ys, color='black', linestyle='dotted')

        # primary
        for name, primary in pred_paths.items():
            xs = [r.x for r in primary]
            ys = [r.y for r in primary]
            # track
            ax.plot(xs, ys, linestyle='solid', label=name,
                    marker='o', markersize=2.5, zorder=1.9)
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                    linestyle='None', zorder=0.9)
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                    linestyle='None', zorder=0.9)

        # neigh predictions
        if pred_neigh_paths is not None:
            for name, neigh_paths in pred_neigh_paths.items():
                for neigh_path in neigh_paths:
                    xs = [r.x for r in neigh_path]
                    ys = [r.y for r in neigh_path]
                    # track
                    ax.plot(xs, ys, linestyle='solid',
                            marker='o', markersize=2.5, zorder=1.9)
                    # markers
                    ax.plot(xs[0:1], ys[0:1], color='black', marker='x',
                            linestyle='None', zorder=0.9)
                    ax.plot(xs[-1:], ys[-1:], color='black', marker='o',
                            linestyle='None', zorder=0.9)

        # frame
        ax.legend()
