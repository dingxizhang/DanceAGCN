import numpy as np
import matplotlib.backends.backend_agg as plt_backend_agg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg


def plot_confusion_matrix(cm, label_index_to_str=None, normalise=True, cmap=plt.cm.Blues, sort_by_n_instances=False):
    instances_per_class = np.sum(cm, axis=1)

    if sort_by_n_instances:
        sort_idx = np.flip(np.argsort(instances_per_class), axis=0)
        instances_per_class = instances_per_class[sort_idx]
        cm = cm[sort_idx, :]

        for r in range(cm.shape[0]):
            cm[r] = cm[r, sort_idx]
    else:
        sort_idx = range(len(instances_per_class))

    if label_index_to_str is None:
        y_labels = ['{} ({})'.format(s, instances_per_class[idx]) for idx, s in enumerate(sort_idx)]
    else:
        # the 'NotFound' is only to allow running experiments on subsets of the dataset
        index_to_str = [label_index_to_str[s_idx] if s_idx in label_index_to_str else 'NotFound!' for s_idx in sort_idx]
        y_labels = ['{}:{} ({})'.format(idx, s, instances_per_class[idx]) for idx, s in enumerate(index_to_str)]

    fig = plt.figure(dpi=150)

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(y_labels))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks, y_labels)

    try:
        plt.tight_layout()
    except Exception:
        pass

    return fig


def figure_to_img(figure, close_figure=True):
    """
    Convert a matplotlib figure to an image

    :param figure: Matplotlib image
    :param close_figure: if True closes the figure
    :return: the image as NumPy array
    """
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image = data.reshape([h, w, 4])[:, :, 0:3]

    if close_figure:
        plt.close(figure)

    return image


def remove_ticks_and_labels(ax, is_3d=True):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    if is_3d:
        ax.set_zticks([])
        ax.set_zlabel(None)


def scale_3d_axis(ax, x_scale=1.0, y_scale=1.0, z_scale=1.0):
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj


def squeeze_subplots(fig):
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)


def create_3d_figure(nrows=1, ncols=1, subplots=1, dpi=150):
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(nrows, ncols, subplots, projection='3d')
    return fig, ax


def set_3d_axis_limits(ax, x_lim=(-1, 1), y_lim=(-1, 1), z_lim=(-1, 1), from_data=None):
    if from_data is not None:
        min_ = from_data.min(axis=0)
        max_ = from_data.max(axis=0)
        x_lim = (min_[0], max_[0])
        y_lim = (min_[2], max_[2])  # sic, because of zdir='y'
        z_lim = (min_[1], max_[1])

    ax.set_xlim3d(x_lim)
    ax.set_ylim3d(y_lim)
    ax.set_zlim3d(z_lim)


def set_2d_axis_limits(ax, x_lim=(-1, 1), y_lim=(-1, 1), from_data=None):
    if from_data is not None:
        min_ = from_data.min(axis=0)
        max_ = from_data.max(axis=0)
        x_lim = (min_[0], max_[0])
        y_lim = (min_[1], max_[1])

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


class KinectsHeatmap:
    def __init__(self, img_path):
        # img_path should be taken from https://vvvv.org/documentation/kinect
        # https://vvvv.org/sites/default/files/imagecache/large/images/kinectskeleton-map2.png
        self.img = mpimg.imread(img_path)
        self.skel_map = {0: {'x': 478, 'y': 347, 'xr': 310.35353535353545, 'yr': 299.3340548340549},
                         1: {'x': 479, 'y': 405, 'xr': 311.39971139971146, 'yr': 238.65584415584425},
                         2: {'x': 481, 'y': 503, 'xr': 313.4920634920636, 'yr': 136.13059163059165},
                         3: {'x': 478, 'y': 532, 'xr': 310.35353535353545, 'yr': 108.79148629148642},
                         4: {'x': 525, 'y': 477, 'xr': 359.52380952380963, 'yr': 163.3311688311689},
                         5: {'x': 529, 'y': 410, 'xr': 363.7085137085138, 'yr': 233.424963924964},
                         6: {'x': 532, 'y': 352, 'xr': 366.84704184704196, 'yr': 294.1031746031747},
                         7: {'x': 536, 'y': 336, 'xr': 371.03174603174614, 'yr': 310.8419913419914},
                         8: {'x': 436, 'y': 475, 'xr': 266.41414141414145, 'yr': 165.42352092352098},
                         9: {'x': 432, 'y': 404, 'xr': 262.2294372294373, 'yr': 239.70202020202026},
                         10: {'x': 427, 'y': 348, 'xr': 256.9985569985571, 'yr': 298.28787878787887},
                         11: {'x': 424, 'y': 332, 'xr': 253.8600288600289, 'yr': 315.02669552669556},
                         12: {'x': 505, 'y': 348, 'xr': 338.60028860028865, 'yr': 298.28787878787887},
                         13: {'x': 505, 'y': 252, 'xr': 338.60028860028865, 'yr': 398.72077922077926},
                         14: {'x': 506, 'y': 181, 'xr': 339.64646464646466, 'yr': 472.99927849927855},
                         15: {'x': 509, 'y': 150, 'xr': 342.7849927849928, 'yr': 505.430735930736},
                         16: {'x': 454, 'y': 348, 'xr': 285.2453102453103, 'yr': 298.28787878787887},
                         17: {'x': 455, 'y': 254, 'xr': 286.2914862914863, 'yr': 396.6284271284272},
                         18: {'x': 455, 'y': 182, 'xr': 286.2914862914863, 'yr': 471.9531024531025},
                         19: {'x': 446, 'y': 151, 'xr': 276.87590187590195, 'yr': 504.38455988455996},
                         20: {'x': 480, 'y': 487, 'xr': 312.4458874458875, 'yr': 152.86940836940846},
                         21: {'x': 525, 'y': 319, 'xr': 359.52380952380963, 'yr': 328.6269841269842},
                         22: {'x': 523, 'y': 332, 'xr': 357.4314574314575, 'yr': 315.02669552669556},
                         23: {'x': 433, 'y': 312, 'xr': 263.2756132756133, 'yr': 335.9502164502165},
                         24: {'x': 436, 'y': 329, 'xr': 266.41414141414145, 'yr': 318.1652236652237}}

    def plot(self, heatmap, nodes=tuple(range(25)), ax=None, node_size=15, cmap='coolwarm', crop=True):
        if ax is None:
            fig, ax = plt.subplots(dpi=150)

        if crop:
            xbounds = [235, 390]
            ybounds = [60, 520]
            ax.imshow(self.img[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1], :])
        else:
            ax.imshow(self.img)

        norm = plt.Normalize(heatmap.min(), heatmap.max())

        x = []
        y = []
        c = []

        for n in nodes:
            node = self.skel_map[n]
            xn = node['xr']
            yn = node['yr']

            if crop:
                xn -= xbounds[0]
                yn -= ybounds[0]

            x.append(xn)
            y.append(yn)
            c.append(heatmap[n])

        ax.scatter(x, y, s=node_size, c=c, norm=norm, cmap=cmap)
        ax.axis('off')

    def highlight_node(self, node, colour='gold', crop=True, plot_img=False, ax=None, node_size=15):
        if ax is None:
            fig, ax = plt.subplots(dpi=150)

        if crop:
            xbounds = [235, 390]
            ybounds = [60, 520]

            if plot_img:
                ax.imshow(self.img[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1], :])
        elif plot_img:
            ax.imshow(self.img)

        node = self.skel_map[node]
        xn = node['xr']
        yn = node['yr']

        if crop:
            xn -= xbounds[0]
            yn -= ybounds[0]

        ax.scatter(xn, yn, s=node_size, c=colour)
        ax.axis('off')
