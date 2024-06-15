import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


import imgviz
COLOR_MAPPING = {
    # 255: (255, 255, 255),
    6: (0, 0, 63),
    9: (0, 191, 127),
    1: (0, 63, 0),
    7: (0, 63, 127),
    8: (0, 63, 191),
    3: (0, 63, 255),
    2: (0, 127, 63),
    5: (0, 127, 127),
    4: (0, 0, 127),
    14: (0, 0, 191),
    13: (0, 0, 255),
    11: (0, 63, 63),
    10: (0, 127, 191),
    0: (0, 127, 255),
    # 0: (255,255,255), 
    12: (0, 100, 155),
    15: (64, 191, 127),
    16: (64, 0, 191),
    17: (128, 63, 63),
    18: (128, 0, 63),
    19: (191, 63, 0),
    20: (255, 127, 0),
    21: (63, 0, 0),
    22: (127, 63, 0),
    23: (63, 255, 0),
    24: (0, 127, 0),
    25: (127, 127, 0),
    26: (63, 0, 63),
    27: (63, 127, 0),
    28: (63, 191, 0),
    29: (191, 127, 0),
    30: (127, 191, 0),
    31: (63, 63, 0),
    32: (100, 155, 0),
    33: (0, 255, 0),
    34: (0, 191, 0),
    35: (191, 127, 64),
    36: (0, 191, 64),
    37: (251, 28, 28),
    38: (47, 246, 57),
    39: (110, 248, 22),
    40: (17, 242, 127),
    41: (255,255,255),
    }
LandDiscoverMapping = [list(COLOR_MAPPING[i]) for i in sorted(COLOR_MAPPING)]
FAST_CLASS = ["A220","A321","A330","A350","ARJ21","Baseball-Field","Basketball-Court",
"Boeing737","Boeing747","Boeing777","Boeing787","Bridge","Bus","C919","Cargo-Truck",
"Dry-Cargo-Ship","Dump-Truck","Engineering-Ship","Excavator","Fishing-Boat",
"Football-Field","Intersection","Liquid-Cargo-Ship","Motorboat","other-airplane",
"other-ship","other-vehicle","Passenger-Ship","Roundabout","Small-Car","Tennis-Court",
"Tractor","Trailer","Truck-Tractor","Tugboat","Van","Warship","background"]
FLAIR_CLASSES = ["building","pervious surface","impervious surface","bare soil",
                 "water","coniferous","deciduous","brushwood","vineyard",
                 "herbaceous vegetation","agricultural land","plowed land","other"]
POTSDAM_CLASS = ['impervious surface', 'building', 'low vegetation', 'tree','car','clutter']
POTSDAM_PALLETE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]
GenericMapping = [list(COLOR_MAPPING[i]) for i in sorted(COLOR_MAPPING)]
FLAIR_PALLETE = GenericMapping

FLOODNET_CLASSES = [
    'Background',
    'building-flooded',
    'building-non-flooded',
    'road-flooded',
    'road-non-flooded',
    'water',
    'tree',
    'vehicle',
    'pool',
    'grass'
]
FLOOD_NET_PALLETE  = [[0, 0, 0],[125,0,0], [255, 0, 0], [255, 255, 0],[0,125,0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128], [255,255,255]]
def label2rgb(image,mask,gt,ds_name):
    image = image.transpose((1,2,0))
    
    image = Image.fromarray(image)
    # gt = Image.fromarray(gt)
    # data = imgviz.data.voc()

    # rgb = data["rgb"]
    # label = data["class_label"]
    rgb = image
    # print(type(rgb))
    array_rgb = np.array(rgb)


    #print(image.shape)
    #print(rgb.shape)

    label = np.array(mask)

    # gt = np.uint8(gt)

    if "FAST_all" in ds_name:
        CLASS = FAST_CLASS
        COLORMAP = LandDiscoverMapping
    elif "potsdam_all" in ds_name:
        CLASS = POTSDAM_CLASS
        COLORMAP = POTSDAM_PALLETE
    # f "FLAIR" in ds_name:
    elif "FLAIR_test" in ds_name:
        CLASS = FLAIR_CLASSES
        COLORMAP = FLAIR_PALLETE
    elif "FloodNet" in ds_name:
        CLASS = FLOODNET_CLASSES
        COLORMAP = FLOOD_NET_PALLETE
    else:
        print("got wrong ds name in visualizer")
        assert(False)
    label_names = [
        "{}:{}".format(i, n) for i, n in enumerate(CLASS)
    ]
    
    # print(gt.max())
    
    FontSize = 20
    # print(label_names)
    label = np.uint8(label)

    isaid_colormap = np.uint8(COLORMAP)
    #print(label.shape)
    #print(isaid_colormap.shape)
    labelviz_withname1 = imgviz.label2rgb(
        label, label_names=label_names, font_size=FontSize, loc="centroid", colormap=isaid_colormap
    )
    labelviz_withname2 = imgviz.label2rgb(
        label, label_names=label_names, font_size=FontSize, loc="rb", colormap=isaid_colormap
    )
    img = imgviz.color.rgb2gray(array_rgb)
    labelviz_withimg = imgviz.label2rgb(label=label, alpha=0.5, image=img,font_size=FontSize, colormap=isaid_colormap)
    gt[np.where(gt==255)] = 37
    gtlabelviz_withname = imgviz.label2rgb(
        label = gt, label_names=label_names, font_size=FontSize, colormap=isaid_colormap, loc="rb"
    )
    # -------------------------------------------------------------------------

    plt.figure(dpi=600)
    
    plt.subplot(151)
    plt.title("img",fontsize=8)
    plt.imshow(array_rgb)
    plt.axis("off")
    
    plt.subplot(152)
    plt.title("pred+img",fontsize=8)
    plt.imshow(labelviz_withimg)
    plt.axis("off")

    plt.subplot(153)
    plt.title("pred.loc=centroid",fontsize=8)
    plt.imshow(labelviz_withname1)
    plt.axis("off")

    plt.subplot(154)
    plt.title("pred.loc=rb",fontsize=8)
    plt.imshow(labelviz_withname2)
    plt.axis("off")

    plt.subplot(155)
    plt.title("gt.loc=rb",fontsize=8)
    plt.imshow(gtlabelviz_withname)
    plt.axis("off")
    
    
    
    
    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img

def save_visual(img,out_map,gt,save_path,ds_name):
    # h, w = img.shape[-2:]
    # gt = gt.resize((256,256))
    gt = np.array(gt)
    # img = np.uint8(img)
    # img = np.uint8(img.squeeze(0).permute(1,2,0).detach().cpu())

    img_out = label2rgb(img,out_map,gt,ds_name)
    plt.imshow(img_out)
    plt.axis("off")
    plt.margins(0,0)
    # plt.savefig(Save_Folder_Path+'diy_outs_RS_'+image_path.split('.')[-2].split('/')[-1]+'.png',dpi=300,bbox_inches='tight', pad_inches = 0.5)
    # plt.savefig(Save_Folder_Path+'diy_outs_RS_debug_'+image_path.split('.')[-2].split('/')[-1]+'.png',dpi=300,bbox_inches='tight', pad_inches = 0.5)
    #
    plt.savefig(save_path,dpi=300,bbox_inches='tight', pad_inches = 0.5)
