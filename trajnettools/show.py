from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc

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
def predicted_paths(input_paths, pred_paths, output_file=None):
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

# def center(path, neigh_path, time_param=(9, 21, 9, 3)):
#     ## Centre scene
#     T_OBS, T_SEQ, T_INT, T_STR = time_param 
#     center = path[T_INT, :]
#     path = path - center
#     neigh_path = neigh_path - center
#     return path, neigh_path

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
def interaction_path(path, neigh, kf=None, output_file=None):
    """Context to plot paths."""
    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax
        
        # Center
        # path, neigh = center(path, neigh)
        center = path[9, :]
        path = path - center
        neigh = neigh - center

        # Primary Track
        ax.scatter(path[:, 0], path[:, 1], color='b', label = 'primary')
        # ax.plot(path[:, 0], path[:, 1], color='b', label = 'primary')
        ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label = 'start point')
        ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label = 'end point')

        # Neighbour Track
        for j in range(neigh.shape[1]):             
            ax.plot(neigh[:, j, 0], neigh[:, j, 1], color='g', label = 'neighbour' + str(j+1))
            ax.plot(neigh[0, j, 0], neigh[0, j, 1], color='g', marker='o')
            ax.plot(neigh[-1, j, 0], neigh[-1, j, 1], color='r', marker='x')

        # kalman if present
        if kf is not None:
            kf = kf - center
            ax.plot(kf[:, 0, 0], kf[:, 0, 1], color='r', label = 'kalman')

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


def makeDynamicPlot(input_paths, path='', scenes_pred=[], draw_triangle=0, centre=True, np=False):
    
    if path == '':
        path = r'figure/dyn_pred_fig.gif'        
    # len_pred = len(scenes_pred)
    
    # scenes = scenes[ind]
    # if len(scenes_pred) > 0:
    #     scenes_pred = scenes_pred[ind][0]
    #     len_pred = 1
    len_pred = 0
    trajectory_i = input_paths[0]
    interaction = input_paths[1:]
    id_tmp = len(interaction)
    print(id_tmp)
    ## Initialization of the plot#
    fig = plt.figure(figsize=(12, 12))
    # lim_sup = 0
    # for i in trajectory_i:
    #     tmp = max(abs(i.x), abs(i.y))
    #     if tmp > lim_sup:
    #         lim_sup = tmp
    # ax1 = plt.axes(xlim=(my_axis[0], my_axis[1]), ylim=(my_axis[2], my_axis[3]))
    ax1 = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    #ax1 = plt.axes()
    line, = ax1.plot([], [], marker='+')

    surround={}
    for w in range(id_tmp):
        surround[w] = interaction[w]

    # Creation of a dictionary with all interacting trajectories
    x_int, y_int = [], []
    x_pred, y_pred = [], []
    dictio = {}
    for j in range(id_tmp):
        dictio['x%s' % j] = []
        dictio['y%s' % j] = []

    ## Plot initialization #2
    lines = []
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for index in range(1 + id_tmp + draw_triangle + len_pred):
        if index < 1:
            lobj = ax1.plot([], [], marker='o', color='b', alpha=1)[0]
        elif index < 1 + id_tmp:
            lobj = ax1.plot([], [], marker='.', color = 'k', alpha = 0.6)[0]
        elif index < 1 + id_tmp + draw_triangle:
            lobj = ax1.plot([], [], marker='.', color = 'g', alpha = 0.6)[0]
        else:
            lobj = ax1.plot([], [], marker='o', color = 'tab:blue', alpha = 1)[0]
        lines.append(lobj)

    # Initialization
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # Create the animation
    def animate(i):
        label = 'timestep {0}'.format(i)
        if not np:
            frame_nb = trajectory_i[i].frame
            xbis = trajectory_i[i].x
            ybis = trajectory_i[i].y
        else:
            xbis = trajectory_i[i][0]
            ybis = trajectory_i[i][1]
        x_int.append(xbis)
        y_int.append(ybis)
        if len(x_int)>5:
            x_int.pop(0)
            y_int.pop(0)
            
        # if len(scenes_pred) > 0:
        #     if i >= 8:
        #         x_pred.append(scenes_pred[i].x)
        #         y_pred.append(scenes_pred[i].y)
        #         if len(x_pred)>3:
        #             x_pred.pop(0)
        #             y_pred.pop(0)

        for j in range(id_tmp):
            if len(dictio['x%s' % j])>4:
                        dictio['x%s' % j].pop(0)
                        dictio['y%s' % j].pop(0)
            if not np:
                for jj in range(len(surround[j])):
                    if surround[j][jj].frame == frame_nb:
                        dictio['x%s' % j].append(surround[j][jj].x)
                        dictio['y%s' % j].append(surround[j][jj].y)
            else: 
                dictio['x%s' % j].append(surround[j][i][0])
                dictio['y%s' % j].append(surround[j][i][1])
        xlist = [x_int]
        ylist = [y_int]
        for j in range(id_tmp):
            xlist.append(dictio['x%s' % j])
            ylist.append(dictio['y%s' % j])

        
        # xlist.append(x_pred)
        # ylist.append(y_pred)

        ax1.set_xlabel(label)

        # print(len(xlist))
        # print(len(lines))
        for lnum, line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.

        return lines

    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=21, blit=True)#interval=500,

    # Save animation
    Writer = animation.writers['imagemagick']
    writer = Writer(fps = 2)
    
    if path is not None:
        anim.save(path, writer=writer, dpi=128)
    #plt.close()
