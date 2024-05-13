import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import get_path_collection_extents
import numpy as np
from PIL import Image, ImageDraw
import string


#plot creation
def getbb(sc, ax):
    ax.figure.canvas.draw()
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            paths = [paths[0]]*len(offsets)
        if len(transforms) < len(offsets):
            transforms = [transforms[0]]*len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t],
                [o], transOffset.frozen())
            bboxes.append(result)

    return bboxes

def create_scatter_plot(x_points, y_points, marker_size=5, marker_style="o", line_style="-", line_width=1, fig_dpi=200,
                        points_color=None, title=None, xlabel_text=None, ylabel_text=None,
                        filename_img="test.jpg"):
    if points_color is None:
        points_color = matplotlib.colormaps['hsv'](np.random.randint(0, 255))
    dct = dict()
    fig, ax = plt.subplots()
    fig.set_dpi(fig_dpi)
    fig.set_tight_layout(True)
    x_min, x_max = x_points.min(), x_points.max()
    x_step = (x_max - x_min) / min(len(x_points), 5)
    y_min, y_max = y_points.min(), y_points.max()
    y_step = (y_max - y_min) / min(len(y_points), 7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.4f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.axis([x_min - x_step, x_max + x_step, y_min - y_step, y_max + y_step])
    ax.set_xticks(np.arange(x_min - np.random.uniform(0, x_step), x_max + np.random.uniform(0, x_step), x_step + np.random.uniform(-x_step / 8, x_step / 4)))
    ax.set_yticks(np.arange(y_min - np.random.uniform(0, y_step), y_max + np.random.uniform(0, y_step), y_step + np.random.uniform(-y_step / 8, y_step / 4)))
    if title is not None:
        ax.set_title(title)
    if xlabel_text is not None:
        ax.set_xlabel(xlabel_text)
    if ylabel_text is not None:
        ax.set_ylabel(ylabel_text)
    # fig.tight_layout()
    points = ax.scatter(x_points, y_points, marker=marker_style, s=marker_size**2, color=points_color, linewidths=0)
    ax.plot(x_points, y_points, marker=marker_style, ms=marker_size, color=points_color, lw=line_width, ls=line_style)
    width, height = fig.canvas.get_width_height()
    fig.canvas.draw()

    dct["plot"] = dict()
    dct["plot"]["width"] = width
    dct["plot"]["height"] = height
    dct["plot"]["dpi"] = fig.dpi
    axes_bbox = ax.get_tightbbox()
    dct["plot"]["axes_coords_px"] = ((axes_bbox.x0, height - axes_bbox.y1), (axes_bbox.x1, height - axes_bbox.y0))
    dct["plot"]["type"] = "scatter"

    if title is not None:
        dct["title"] = dict()
        dct["title"]["text"] = ax.get_title()
        title_bbox = ax.title.get_window_extent()
        dct["title"]["coords_px"] = ((title_bbox.x0, height - title_bbox.y1), (title_bbox.x1, height - title_bbox.y0))

    dct["points"] = dict()
    dct["points"]["coords_px"] = list()
    for bbox in getbb(points, ax):
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["points"]["coords_px"].append(point_coords)
    dct["points"]["vals"] = list(zip(x_points.tolist(), y_points.tolist()))

    dct["xticks"] = dict()
    xtickslocs = ax.get_xticks()
    ymin, _ = ax.get_ylim()
    xticks_coords = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
    xticks_coords = [(xtick, height - ytick) for xtick, ytick in xticks_coords]
    dct["xticks"]["coords_px"] = xticks_coords

    dct["yticks"] = dict()
    ytickslocs = ax.get_yticks()
    xmin, _ = ax.get_xlim()
    yticks_coords = ax.transData.transform([(xmin, ytick) for ytick in ytickslocs])
    yticks_coords = [(xtick, height - ytick) for xtick, ytick in yticks_coords]
    dct["yticks"]["coords_px"] = yticks_coords

    dct["xlabels"] = dict()
    dct["xlabels"]["coords_px"] = list()
    dct["xlabels"]["text"] = list()
    for tick in ax.get_xticklabels():
        dct["xlabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["xlabels"]["coords_px"].append(point_coords)

    dct["ylabels"] = dict()
    dct["ylabels"]["coords_px"] = list()
    dct["ylabels"]["text"] = list()
    for tick in ax.get_yticklabels():
        dct["ylabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["ylabels"]["coords_px"].append(point_coords)

    if xlabel_text is not None:
        dct["xlabel"] = dict()
        dct["xlabel"]["text"] = ax.get_xlabel()
        bbox_xlabel = ax.xaxis.label.get_window_extent()
        dct["xlabel"]["coords_px"] = ((bbox_xlabel.x0, height - bbox_xlabel.y1), (bbox_xlabel.x1, height - bbox_xlabel.y0))

    if ylabel_text is not None:
        dct["ylabel"] = dict()
        dct["ylabel"]["text"] = ax.get_ylabel()
        bbox_ylabel = ax.yaxis.label.get_window_extent()
        dct["ylabel"]["coords_px"] = ((bbox_ylabel.x0, height - bbox_ylabel.y1), (bbox_ylabel.x1, height - bbox_ylabel.y0))

    fig.savefig(filename_img, dpi=fig.dpi)
    plt.close(fig)
    return dct


def create_vertical_bar_plot(x_labels, y_points, bar_width=0.8, fig_dpi=200,
                        bars_color=None, title=None, xlabel_text=None, ylabel_text=None,
                        filename_img="test.jpg"):
    if bars_color is None:
        bars_color = matplotlib.colormaps['hsv'](np.random.randint(0, 255))
    dct = dict()
    fig, ax = plt.subplots()
    fig.set_dpi(fig_dpi)
    fig.set_tight_layout(True)
    y_min, y_max = 0, y_points.max()
    y_step = (y_max - y_min) / min(len(y_points), 7)
    ax.axis([-1 + np.random.uniform(0, bar_width / 2), len(y_points) - np.random.uniform(0, bar_width / 2), y_min, y_max + y_step])
    ax.set_xticks(np.arange(len(y_points)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(y_min, y_max + np.random.uniform(0, y_step), y_step + np.random.uniform(-y_step / 8, y_step / 4)))

    if title is not None:
        ax.set_title(title)
    if xlabel_text is not None:
        ax.set_xlabel(xlabel_text)
    if ylabel_text is not None:
        ax.set_ylabel(ylabel_text)
    ind = np.arange(len(y_points))
    bars = ax.bar(ind, y_points, width=bar_width, color=bars_color)
    width, height = fig.canvas.get_width_height()
    fig.canvas.draw()

    dct["plot"] = dict()
    dct["plot"]["width"] = width
    dct["plot"]["height"] = height
    dct["plot"]["dpi"] = fig.dpi
    axes_bbox = ax.get_tightbbox()
    dct["plot"]["axes_coords_px"] = ((axes_bbox.x0, height - axes_bbox.y1), (axes_bbox.x1, height - axes_bbox.y0))
    dct["plot"]["type"] = "vertical_bar"

    if title is not None:
        dct["title"] = dict()
        dct["title"]["text"] = ax.get_title()
        title_bbox = ax.title.get_window_extent()
        dct["title"]["coords_px"] = ((title_bbox.x0, height - title_bbox.y1), (title_bbox.x1, height - title_bbox.y0))


    dct["bars"] = dict()
    dct["bars"]["coords_px"] = list()
    for bar in bars:
        bbox = bar.get_bbox().transformed(ax.transData)
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["bars"]["coords_px"].append(point_coords)
    dct["bars"]["vals"] = y_points.tolist()

    dct["xticks"] = dict()
    xtickslocs = ax.get_xticks()
    ymin, _ = ax.get_ylim()
    xticks_coords = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
    xticks_coords = [(xtick, height - ytick) for xtick, ytick in xticks_coords]
    dct["xticks"]["coords_px"] = xticks_coords

    dct["yticks"] = dict()
    ytickslocs = ax.get_yticks()
    xmin, _ = ax.get_xlim()
    yticks_coords = ax.transData.transform([(xmin, ytick) for ytick in ytickslocs])
    yticks_coords = [(xtick, height - ytick) for xtick, ytick in yticks_coords]
    dct["yticks"]["coords_px"] = yticks_coords

    dct["xlabels"] = dict()
    dct["xlabels"]["coords_px"] = list()
    dct["xlabels"]["text"] = list()
    for tick in ax.get_xticklabels():
        dct["xlabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["xlabels"]["coords_px"].append(point_coords)

    dct["ylabels"] = dict()
    dct["ylabels"]["coords_px"] = list()
    dct["ylabels"]["text"] = list()
    for tick in ax.get_yticklabels():
        dct["ylabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["ylabels"]["coords_px"].append(point_coords)

    if xlabel_text is not None:
        dct["xlabel"] = dict()
        dct["xlabel"]["text"] = ax.get_xlabel()
        bbox_xlabel = ax.xaxis.label.get_window_extent()
        dct["xlabel"]["coords_px"] = ((bbox_xlabel.x0, height - bbox_xlabel.y1), (bbox_xlabel.x1, height - bbox_xlabel.y0))

    if ylabel_text is not None:
        dct["ylabel"] = dict()
        dct["ylabel"]["text"] = ax.get_ylabel()
        bbox_ylabel = ax.yaxis.label.get_window_extent()
        dct["ylabel"]["coords_px"] = ((bbox_ylabel.x0, height - bbox_ylabel.y1), (bbox_ylabel.x1, height - bbox_ylabel.y0))

    fig.savefig(filename_img, dpi=fig.dpi)
    plt.close(fig)
    return dct


def create_horizontal_bar_plot(x_points, y_labels, bar_height=0.8, fig_dpi=200,
                        bars_color=None, title=None, xlabel_text=None, ylabel_text=None,
                        filename_img="test.jpg"):
    if bars_color is None:
        bars_color = matplotlib.colormaps['hsv'](np.random.randint(0, 255))
    dct = dict()
    fig, ax = plt.subplots()
    fig.set_dpi(fig_dpi)
    fig.set_tight_layout(True)
    x_min, x_max = 0, x_points.max()
    x_step = (x_max - x_min) / min(len(x_points), 5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.axis([x_min, x_max + x_step, -1 + np.random.uniform(0, bar_height / 2), len(x_points) - np.random.uniform(0, bar_height / 2)])
    ax.set_yticks(np.arange(len(x_points)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(x_min, x_max + np.random.uniform(0, x_step), x_step + np.random.uniform(-x_step / 8, x_step / 4)))

    if title is not None:
        ax.set_title(title)
    if xlabel_text is not None:
        ax.set_xlabel(xlabel_text)
    if ylabel_text is not None:
        ax.set_ylabel(ylabel_text)
    ind = np.arange(len(x_points))
    bars = ax.barh(ind, x_points, height=bar_height, color=bars_color)
    width, height = fig.canvas.get_width_height()
    fig.canvas.draw()

    dct["plot"] = dict()
    dct["plot"]["width"] = width
    dct["plot"]["height"] = height
    dct["plot"]["dpi"] = fig.dpi
    axes_bbox = ax.get_tightbbox()
    dct["plot"]["axes_coords_px"] = ((axes_bbox.x0, height - axes_bbox.y1), (axes_bbox.x1, height - axes_bbox.y0))
    dct["plot"]["type"] = "horizontal_bar"

    if title is not None:
        dct["title"] = dict()
        dct["title"]["text"] = ax.get_title()
        title_bbox = ax.title.get_window_extent()
        dct["title"]["coords_px"] = ((title_bbox.x0, height - title_bbox.y1), (title_bbox.x1, height - title_bbox.y0))


    dct["bars"] = dict()
    dct["bars"]["coords_px"] = list()
    for bar in bars:
        bbox = bar.get_bbox().transformed(ax.transData)
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["bars"]["coords_px"].append(point_coords)
    dct["bars"]["vals"] = x_points.tolist()

    dct["xticks"] = dict()
    xtickslocs = ax.get_xticks()
    ymin, _ = ax.get_ylim()
    xticks_coords = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
    xticks_coords = [(xtick, height - ytick) for xtick, ytick in xticks_coords]
    dct["xticks"]["coords_px"] = xticks_coords

    dct["yticks"] = dict()
    ytickslocs = ax.get_yticks()
    xmin, _ = ax.get_xlim()
    yticks_coords = ax.transData.transform([(xmin, ytick) for ytick in ytickslocs])
    yticks_coords = [(xtick, height - ytick) for xtick, ytick in yticks_coords]
    dct["yticks"]["coords_px"] = yticks_coords

    dct["xlabels"] = dict()
    dct["xlabels"]["coords_px"] = list()
    dct["xlabels"]["text"] = list()
    for tick in ax.get_xticklabels():
        dct["xlabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["xlabels"]["coords_px"].append(point_coords)

    dct["ylabels"] = dict()
    dct["ylabels"]["coords_px"] = list()
    dct["ylabels"]["text"] = list()
    for tick in ax.get_yticklabels():
        dct["ylabels"]["text"].append(tick.get_text())
        bbox = tick.get_window_extent()
        point_coords = ((bbox.x0, height - bbox.y1), (bbox.x1, height - bbox.y0))
        dct["ylabels"]["coords_px"].append(point_coords)

    if xlabel_text is not None:
        dct["xlabel"] = dict()
        dct["xlabel"]["text"] = ax.get_xlabel()
        bbox_xlabel = ax.xaxis.label.get_window_extent()
        dct["xlabel"]["coords_px"] = ((bbox_xlabel.x0, height - bbox_xlabel.y1), (bbox_xlabel.x1, height - bbox_xlabel.y0))

    if ylabel_text is not None:
        dct["ylabel"] = dict()
        dct["ylabel"]["text"] = ax.get_ylabel()
        bbox_ylabel = ax.yaxis.label.get_window_extent()
        dct["ylabel"]["coords_px"] = ((bbox_ylabel.x0, height - bbox_ylabel.y1), (bbox_ylabel.x1, height - bbox_ylabel.y0))

    fig.savefig(filename_img, dpi=fig.dpi)
    plt.close(fig)
    return dct


#generate data for title and points
def create_bag_of_words():
    with open("random_words.txt") as file:
        data = file.read()
        data = data.translate(str.maketrans('', '', string.punctuation))
        data = data.translate(str.maketrans('', '', string.digits))
        data = list(set(map(str.strip, data.split())))
        data = list(filter(lambda x: len(x) < 15, data))
    return data

words = create_bag_of_words()
short_words = list(filter(lambda x: len(x) < 9, words))

def generate_text():
    num_of_words = np.random.randint(0, 6)
    if num_of_words == 0:
        return None
    elif num_of_words <= 3:
        return ' '.join(np.random.choice(words, num_of_words))
    else:
        line1 = ' '.join(np.random.choice(words, 3))
        line2 = ' '.join(np.random.choice(words, num_of_words - 3))
        return line1 + "\n" + line2

def generate_linear():
    left_border = np.random.uniform(-1000, 500)
    right_border = np.random.uniform(left_border, left_border + 500)
    k = np.random.uniform(-50, 50)
    b = np.random.uniform(-500, 500)
    points_amount = np.random.randint(5, 20)
    x = np.random.uniform(left_border, right_border, points_amount)
    y = k * x + b
    return x, y

def generate_quadratic():
    left_border = np.random.uniform(-50, 50)
    right_border = np.random.uniform(left_border, left_border + 50)
    a = np.random.uniform(-3, 3)
    b = np.random.uniform(-5, 5)
    c = np.random.uniform(-500, 500)
    points_amount = np.random.randint(5, 20)
    x = np.random.uniform(left_border, right_border, points_amount)
    x.sort()
    y = a * (x ** 2) + b * x + c
    return x, y

def generate_cubic():
    left_border = np.random.uniform(-5, 0)
    right_border = np.random.uniform(0, -left_border)
    a = np.random.uniform(-2, 2)
    b = np.random.uniform(-5, 5)
    c = np.random.uniform(-10, 10)
    d = np.random.uniform(-100, 100)
    points_amount = np.random.randint(5, 20)
    x = np.random.uniform(left_border, right_border, points_amount)
    x.sort()
    y = a * (x ** 3) + b * (x ** 2) + c * x + d
    return x, y

def generate_trig():
    left_border = np.random.uniform(-4, 0)
    right_border = np.random.uniform(0, -left_border)
    points_amount = np.random.randint(5, 20)
    a = np.random.uniform(-5, 5)
    trig_fun = np.random.choice([np.sin, np.cos, np.tan])
    x = np.random.uniform(left_border, right_border, points_amount)
    x.sort()
    y = a * trig_fun(x)
    return x, y

def generate_random():
    left_border = np.random.uniform(-1000, 0)
    right_border = np.random.uniform(0, -left_border)
    points_amount = np.random.randint(5, 20)
    x = np.random.uniform(left_border, right_border, points_amount)
    x.sort()
    ys = [np.random.uniform(-10000, 10000)]
    ys.append(ys[-1] + np.random.uniform(-100, 100))
    while len(ys) < points_amount:
        if np.random.random() < 0.7:
            big_diff = np.random.uniform(0, 10)
            ys.append(ys[-1] + np.sign(ys[-1] - ys[-2]) * big_diff)
        else:
            small_diff = np.random.uniform(-1, 0)
            ys.append(ys[-1] + np.sign(ys[-1] - ys[-2]) * small_diff)
    y = np.array(ys)
    return x, y

scatter_function_types = [generate_linear, generate_quadratic, generate_cubic, generate_trig, generate_random]

def generate_scatter_data():
    function_type = np.random.choice(scatter_function_types)
    return function_type()

def generate_bar_data():
    bar_amount = np.random.randint(3, 8)
    x_labels = np.random.choice(short_words, bar_amount).tolist()
    y_points = np.random.uniform(0, 10000, bar_amount)
    return x_labels, y_points


#create image
plot_types = ["vertical_bar", "horizontal_bar", "scatter"]
def create_plot(max_dpi=220):
    graph_type = np.random.choice(plot_types)
    if graph_type == "vertical_bar":
        x_labels, y_points = generate_bar_data()
        bar_width = np.random.uniform(0.2, 1)
        title = generate_text()
        xlabel_text = generate_text()
        ylabel_text = generate_text()
        fig_dpi = np.random.randint(100, max_dpi)
        dct = create_vertical_bar_plot(x_labels, y_points, bar_width=bar_width, fig_dpi=fig_dpi,
                                    title=title, xlabel_text=xlabel_text, ylabel_text=ylabel_text, filename_img="temp.jpg")
    elif graph_type == "horizontal_bar":
        y_labels, x_points = generate_bar_data()
        bar_height = np.random.uniform(0.2, 1)
        title = generate_text()
        xlabel_text = generate_text()
        ylabel_text = generate_text()
        fig_dpi = np.random.randint(100, max_dpi)
        dct = create_horizontal_bar_plot(x_points, y_labels, bar_height=bar_height, fig_dpi=fig_dpi,
                                    title=title, xlabel_text=xlabel_text, ylabel_text=ylabel_text, filename_img="temp.jpg")
    elif graph_type == "scatter":
        marker_styles = []
        line_styles = [" ", "-", "--", "-.", ":"]
        marker_styles = ["o", "^", "p", "*", "h", "s"]
        x, y = generate_scatter_data()
        marker_size = np.random.uniform(2, 6)
        marker_style = np.random.choice(marker_styles)
        line_style = np.random.choice(line_styles)
        line_width = np.random.uniform(0.4, 1)
        title = generate_text()
        xlabel_text = generate_text()
        ylabel_text = generate_text()
        fig_dpi = np.random.randint(100, max_dpi)
        dct = create_scatter_plot(x, y, fig_dpi=fig_dpi, marker_size=marker_size, marker_style=marker_style, line_style=line_style, line_width=line_width,
                                title=title, xlabel_text=xlabel_text, ylabel_text=ylabel_text, filename_img="temp.jpg")
    else:
        raise ValueError("No such plot type")
    return dct

EPS = 5

def update_tuple_point(point, x_start, y_start):
    return ((point[0][0] + x_start, point[0][1] + y_start), (point[1][0] + x_start, point[1][1] + y_start))

def update_dct(dct, x_start, y_start):
    dct["plot"]["axes_coords_px"] = update_tuple_point(dct["plot"]["axes_coords_px"], x_start, y_start)
    keys_ticks = ["xticks", "yticks"]
    for key in keys_ticks:
        for i in range(len(dct[key]["coords_px"])):
            dct[key]["coords_px"][i] = ((dct[key]["coords_px"][i][0] - EPS + x_start, dct[key]["coords_px"][i][1] - EPS + y_start), (dct[key]["coords_px"][i][0] + EPS + x_start, dct[key]["coords_px"][i][1] + EPS + y_start))
    
    keys = ["title", "xlabel", "ylabel"]
    for key in keys:
        if key in dct:
            dct[key]["coords_px"] = update_tuple_point(dct[key]["coords_px"], x_start, y_start)
    # keys_lst = ["points", "bars", "xticks", "xlabels", "yticks", "ylabels"]
    keys_lst = ["points", "bars", "xlabels", "ylabels"]
    for key in keys_lst:
        if key in dct:
            for i in range(len(dct[key]["coords_px"])):
                dct[key]["coords_px"][i] = update_tuple_point(dct[key]["coords_px"][i], x_start, y_start)
    dct["plot"]["coords_px"] = ((x_start, y_start), (x_start + dct["plot"]["width"], y_start + dct["plot"]["height"]))
    return dct
    

def create_img(img_path="result.jpg", plot_amount=None):
    width, height = 1920, 1920
    image = Image.new('RGB', (width, height), color=(255,255,255))
    if plot_amount is None:
        plot_amount = np.random.randint(1, 3)
    dcts = []
    if plot_amount == 1:
        dct = create_plot()
        plot_image = Image.open("temp.jpg")
        x_start = np.random.randint(1, width - plot_image.width)
        y_start = np.random.randint(1, height - plot_image.height)
        dct = update_dct(dct, x_start, y_start)
        dcts.append(dct)
        image.paste(plot_image, (x_start, y_start))
        image.save(img_path)
    elif plot_amount == 2:
        dct = create_plot(max_dpi=140)
        plot_image = Image.open("temp.jpg")
        x_start = np.random.randint(1, width / 2 - plot_image.width)
        y_start = np.random.randint(1, height - plot_image.height)
        dct = update_dct(dct, x_start, y_start)
        dcts.append(dct)
        image.paste(plot_image, (x_start, y_start))
        dct = create_plot(max_dpi=140)
        old_width = plot_image.width
        plot_image = Image.open("temp.jpg")
        x_start = np.random.randint(x_start + old_width + 1, width - plot_image.width)
        y_start = np.random.randint(1, height - plot_image.height)
        dct = update_dct(dct, x_start, y_start)
        dcts.append(dct)
        image.paste(plot_image, (x_start, y_start))
        image.save(img_path)
    else:
        raise ValueError("Incorrect plot amount")
    return dcts