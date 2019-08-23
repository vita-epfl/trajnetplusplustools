from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt


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

def theta_rotation(xy, theta):
    # rotates scene by theta
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)


def center_scene(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Centre scene
    T_OBS, T_SEQ, T_INT, T_STR = time_param 
    center = path[T_INT, :]
    path = path - center
    neigh_path = neigh_path - center

    k = T_INT - T_STR
    while sum(path[k, :]==0)==2 and k > 0:
        k -= 1
    if k > 0:
        thet = np.pi + np.arccos((path[k, 1])/(np.linalg.norm([0, -1])*np.linalg.norm(path[k, :])))
        if path[k, 0] < 0:
            thet = -thet
        norm_path = theta_rotation(path[:, np.newaxis, :], thet)
        norm_neigh_path = theta_rotation(neigh_path, thet)
        return norm_path[:, 0], norm_neigh_path
    else:
        return path, neigh_path

def center(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Centre scene
    T_OBS, T_SEQ, T_INT, T_STR = time_param 
    center = path[T_INT, :]
    path = path - center
    neigh_path = neigh_path - center
    return path, neigh_path

@contextmanager
def centre_paths(input_paths, output_file=None):
    """Context to plot centred paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10) 

        yield ax

        c_x = input_paths[0][8].x
        c_y = input_paths[0][8].y

        # primary
        xs = [r.x - c_x for r in input_paths[0]]
        ys = [r.y - c_y for r in input_paths[0]]

        # track
        ax.plot(xs, ys, color='red', linestyle='solid', label='primary',
                marker='o', markersize=2.5, zorder=1.9)
        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start',
                linestyle='None', zorder=0.9)
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end',
                linestyle='None', zorder=0.9)

        # other tracks
        for ped_rows in input_paths[1:]:
            xs = [r.x - c_x for r in ped_rows]
            ys = [r.y - c_y for r in ped_rows]
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
            # track
            ax.plot(xs, ys, color='blue', linestyle='dotted')

        # frame
        ax.legend()

@contextmanager
def interaction_path(path, neigh, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax
        
        # Center
        path, neigh = center(path, neigh)

        # Primary Track
        ax.plot(path[:, 0], path[:, 1], label = 'primary')
        ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label = 'start point')
        ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label = 'end point')

        # Neighbour Track
        for j in range(neigh.shape[1]):
            ax.plot(neigh[:, j, 0], neigh[:, j, 1], label = 'neighbour' + str(j+1))
            ax.plot(neigh[0, j, 0], neigh[0, j, 1], color='g', marker='o', label = 'start point')
            ax.plot(neigh[-1, j, 0], neigh[-1, j, 1], color='r', marker='x', label = 'end point')

        # frame
        ax.legend()

## Created a separate function to modify groups and interactions individually
@contextmanager
def group_path(path, group, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        # path, neigh = center_scene(path, neigh)
        ax.plot(path[:, 0], path[:, 1], label = 'primary')
        ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label = 'start point')
        ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label = 'end point')

        for j in range(group.shape[1]):
            ax.plot(group[:, j, 0], group[:, j, 1], label = 'neigh' + str(j+1))
            ax.plot(group[0, j, 0], group[0, j, 1], color='g', marker='o', label = 'start point')
            ax.plot(group[-1, j, 0], group[-1, j, 1], color='r', marker='x', label = 'end point')

        # frame
        ax.legend()