from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from figure_get.forms import UploadFile


from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

model_plot_type_path = "/Users/dgeka24/coursework_django_new/djsite/weights/yolo_plot_type.pt"
model_general_info_path = "/Users/dgeka24/coursework_django_new/djsite/weights/yolo_general_info.pt"
model_bar_path = "/Users/dgeka24/coursework_django_new/djsite/weights/yolo_bars.pt"
model_scatter_path = "/Users/dgeka24/coursework_django_new/djsite/weights/yolo_points.pt"


model_plot_type = YOLO(model_plot_type_path)
model_general_info = YOLO(model_general_info_path)
model_bar = YOLO(model_bar_path)
model_scatter = YOLO(model_scatter_path)

def general_data(image, plot_type):
    preds = model_general_info(image, verbose=False)
    data = {}
    for pred in preds[0]:
        conf = pred.boxes.conf.item()
        if conf < 0.4:
            continue
        cls = round((pred.boxes.cls.item())) # 0: 'title', 1: 'xtitle', 2: 'ytitle', 3: 'xtick', 4: 'ytick', 5: 'xlabel', 6: 'ylabel'
        new_borders = tuple(map(int, pred.boxes.xyxy.tolist()[0]))
        if cls != 2:
            image_cropped = np.array(image.crop(new_borders))
        else:
            image_cropped = np.array(image.crop(new_borders).transpose(Image.ROTATE_270))
        if cls == 0:
            
            image_pil = Image.fromarray(np.uint8(image_cropped)).convert('RGB')
            image_pil = image_pil.resize((image_pil.size[0] * 2, image_pil.size[1] * 2))
            image_pil = np.array(image_pil)
            text = reader.readtext(image_pil)
            if len(text) > 0:
                data['title'] = text[0][1]
        elif cls == 1:
            image_pil = Image.fromarray(np.uint8(image_cropped)).convert('RGB')
            image_pil = image_pil.resize((image_pil.size[0] * 2, image_pil.size[1] * 2))
            image_pil = np.array(image_pil)
            text = reader.readtext(image_pil)
            text = reader.readtext(image_cropped)
            if len(text) > 0:
                data['xtitle'] = text[0][1]
        elif cls == 2:
            image_pil = Image.fromarray(np.uint8(image_cropped)).convert('RGB')
            image_pil = image_pil.resize((image_pil.size[0] * 2, image_pil.size[1] * 2))
            image_pil = np.array(image_pil)
            text = reader.readtext(image_pil)
            if len(text) > 0:
                data['ytitle'] = text[0][1]
        elif cls == 5:
            if "xlabel" not in data:
                data['xlabel'] = []
            image_pil = Image.fromarray(np.uint8(image_cropped)).convert('RGB')
            image_pil = image_pil.resize((image_pil.size[0] * 2, image_pil.size[1] * 2))
            image_pil = np.array(image_pil)
            if plot_type == 0 or plot_type == 2:
                text = reader.readtext(image_pil, allowlist='01234567890-.')
            else:
                text = reader.readtext(image_pil)
            try:
                if len(text) > 0:
                    if plot_type == 0 or plot_type == 2:
                        text = float(text[0][1])
                    else:
                        text = text[0][1]
                    data['xlabel'].append((text, ((new_borders[0] + new_borders[2]) / 2, (new_borders[1] + new_borders[3]) / 2 )))
            except Exception:
                pass
        elif cls == 6:
            if "ylabel" not in data:
                data['ylabel'] = []
            image_pil = Image.fromarray(np.uint8(image_cropped)).convert('RGB')
            image_pil = image_pil.resize((image_pil.size[0] * 2, image_pil.size[1] * 2))
            image_pil = np.array(image_pil)
            if plot_type == 0 or plot_type == 1:
                text = reader.readtext(image_pil, allowlist='01234567890-.')
            else:
                text = reader.readtext(image_pil)
            try:
                if len(text) > 0:
                    if plot_type == 0 or plot_type == 1:
                        text = float(text[0][1])
                    else:
                        text = text[0][1]
                    data['ylabel'].append((text, ((new_borders[0] + new_borders[2]) / 2, (new_borders[1] + new_borders[3]) / 2 )))
            except Exception:
                pass
    return data

def get_y_data(data):
    if 'ylabel' not in data or len(data['ylabel']) < 2:
        return None, None, None
    coord_max = -1000000
    coord_min = 1000000
    y_min = 0
    y_max = 0
    for point in data['ylabel']:
        if point[1][1] > coord_max:
            coord_max = point[1][1]
            y_max = point[0]
        if point[1][1] < coord_min:
            coord_min = point[1][1]
            y_min = point[0]
    y_start = y_min
    y_start_px = coord_min
    y_step = (y_max - y_min) / (coord_max - coord_min)
    return y_start, y_start_px, y_step

def get_x_data(data):
    if 'xlabel' not in data or len(data['xlabel']) < 2:
        return None, None, None
    coord_max = -1000000
    coord_min = 1000000
    x_min = 0
    x_max = 0
    for point in data['xlabel']:
        if point[1][0] > coord_max:
            coord_max = point[1][0]
            x_max = point[0]
        if point[1][0] < coord_min:
            coord_min = point[1][0]
            x_min = point[0]
    x_start = x_min
    x_start_px = coord_min
    x_step = (x_max - x_min) / (coord_max - coord_min)
    return x_start, x_start_px, x_step

def get_scatter_data(image):
    data = general_data(image, 0)
    preds = model_scatter(image, verbose=False)
    
    y_start, y_start_px, y_step = get_y_data(data)
    x_start, x_start_px, x_step = get_x_data(data)
    
    if y_start is None or x_start is None:
        return data
    
    data['points'] = []
    
    for pred in preds[0]:
        conf = pred.boxes.conf.item()
        if conf < 0.5:
            continue
        new_borders = tuple(map(int, pred.boxes.xyxy.tolist()[0]))
        point_coords = ((new_borders[0] + new_borders[2]) / 2, (new_borders[1] + new_borders[3]) / 2)
        data['points'].append((x_start + (point_coords[0] - x_start_px) * x_step, y_start + (point_coords[1] - y_start_px) * y_step))
    if 'xlabel' in data:
        del data['xlabel']
    if 'ylabel' in data:
        del data['ylabel']
    return data

def get_vertical_bar_data(image):
    data = general_data(image, 1)
    y_start, y_start_px, y_step = get_y_data(data)
    if y_start is None:
        return data
    preds = model_bar(image, verbose=False)
    bars_data = []
    for pred in preds[0]:
        conf = pred.boxes.conf.item()
        if conf < 0.5:
            continue
        new_borders = tuple(map(int, pred.boxes.xyxy.tolist()[0]))
        x_pos, y_pos = (new_borders[0] + new_borders[2]) / 2, min(new_borders[1],new_borders[3])
        bars_data.append((x_pos, y_start + (y_pos - y_start_px) * y_step))
    data['bars'] = []
    for xlabel in data['xlabel']:
        x_pos_label = xlabel[1][0]
        x_text_label = xlabel[0]
        dist = 100000
        i = -1
        for idx, bar in enumerate(bars_data):
            if abs(bar[0] - x_pos_label) < dist:
                dist = abs(bar[0] - x_pos_label)
                i = idx
        data['bars'].append((x_text_label, bars_data[i][1]))
    if 'xlabel' in data:
        del data['xlabel']
    if 'ylabel' in data:
        del data['ylabel']
    return data
    
def get_horizontal_bar_data(image):
    data = general_data(image, 2)
    x_start, x_start_px, x_step = get_x_data(data)
    if x_start is None:
        return data
    preds = model_bar(image, verbose=False)
    bars_data = []
    for pred in preds[0]:
        conf = pred.boxes.conf.item()
        if conf < 0.5:
            continue
        new_borders = tuple(map(int, pred.boxes.xyxy.tolist()[0]))
        x_pos, y_pos = max(new_borders[0],new_borders[2]), (new_borders[1] + new_borders[3]) / 2
        bars_data.append((x_start + (x_pos - x_start_px) * x_step, y_pos))
    data['bars'] = []
    for ylabel in data['ylabel']:
        y_pos_label = ylabel[1][1]
        y_text_label = ylabel[0]
        dist = 100000
        i = -1
        for idx, bar in enumerate(bars_data):
            if abs(bar[1] - y_pos_label) < dist:
                dist = abs(bar[1] - y_pos_label)
                i = idx
        data['bars'].append((y_text_label, bars_data[i][0]))
    if 'xlabel' in data:
        del data['xlabel']
    if 'ylabel' in data:
        del data['ylabel']
    return data
    
func_by_cls = [get_scatter_data, get_vertical_bar_data, get_horizontal_bar_data]

def predict_image(image_path):
    background = Image.new('RGB', (1920, 1920), color=(255,255,255))
    image = Image.open(image_path).convert('RGB')
    background.paste(image, (5, 5))
    preds = model_plot_type.predict(background, verbose=False)
    data = []
    for pred in preds[0]:
        cls = round((pred.boxes.cls.item())) # 0-scatter, 1-vertical, 2-horizontal
        box = pred.boxes
        new_borders = tuple(map(int, box.xyxy.tolist()[0]))
        image_cropped = background.crop(new_borders)
        data.append(func_by_cls[cls](image_cropped))
    return data


menu = [{'title': 'Главная страница', 'url_name': 'home'},
        {'title': 'О нас', 'url_name': 'about'}]


def handle_upload_file(file):
    with open(f"uploads/{file.name}", "wb+") as temp:
        for chunk in file.chunks():
            temp.write(chunk)
    image_path = f"uploads/{file.name}"
    data = predict_image(image_path)
    with open(f"res.txt", "w") as temp:
        temp.write(str(data))
    

    

def index(request):
    if request.method == "POST":
        # handle_upload_file(request.FILES["file_upload"])
        form = UploadFile(request.POST, request.FILES)
        if form.is_valid():
            handle_upload_file(form.cleaned_data["file"])
        file_path = "res.txt"
        FilePointer = open(file_path, "r")
        response = HttpResponse(FilePointer,content_type='application/txt')
        response['Content-Disposition'] = 'attachment; filename=results.txt'
        return response
    else:
        form = UploadFile()
    data = {
        'menu': menu,
        'form': form
    }
    return render(request, 'index.html', context=data)


def about(request):
    data = {
        'menu': menu
    }
    return render(request, 'about.html', context=data)


def page_not_found(request, exception):
    return HttpResponseNotFound("<h1>Страница не найдена</h1>")

