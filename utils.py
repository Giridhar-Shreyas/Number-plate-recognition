from xml.dom.minidom import parse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_pose(image, data):
    """
    Args: 
        image (str): Path to the image with the resolution 640x640
        data list[float]: A list of exactly 5 values with format [class, anchor x, anchor y, width, height]
    """

    im = Image.open(image)
    fig, ax = plt.subplots()
    ax.imshow(im)

    for i in data:
        xcent = i[1]*640
        ycent = i[2]*640
        xanc = xcent-((i[3]*640)/2)
        yacn = ycent-((i[4]*640)/2)
        ax.add_patch(patches.Rectangle((xanc,yacn), (i[3]*640), (i[4]*640), linewidth =1, edgecolor = 'r', facecolor='none'))

    plt.show()
    

def test_draw_pose():
    i = 0
    k = 0
    path_img = os.getcwd() + '/data/data_2_yolo_8/train/images'
    path_label = os.getcwd() + '/data/data_2_yolo_8/train/labels'
    for img, label in zip(os.listdir(path_img), os.listdir(path_label)):
        if(i > 10):
            break
        if(k % 3 == 0):
            with open(path_label+'/'+label) as fp:
                data = [list(map(float, line.strip().split(' '))) for line in fp]
            draw_pose(path_img+'/'+img, data)
            i+=1
        k+=1



def convert_Yolo8():

    for i in range(249):
        ii = i+1
        img = os.getcwd() + '/data/data_1/images/N'+str(ii)+'jpeg'
        label = os.getcwd() + '/data/data_1/images/N'+str(ii)+'xml'
        rezise_img(img)

 
        doc = parse(label)
        print(doc.getElementsByTagName("")[0].firstChild.nodeValue)


def rezise_img(imgFile):
    """
    Args:
        imgFile (str): Path to the image needed to be resized 
    """
    img = imgFile
    image = Image.open(img)
    image = image.resize((640,640))
    image.save(imgFile)

def convert_label(label):
    """
    Args:
        label (str): File path to the label in xml format
        key tags: Height, Width, bounding box (xmin, ymin, xmax, ymax) 
    """

    doc = parse(label)
    resHeight = doc.getElementsByTagName("height")[0].firstChild.nodeValue
    resWidth = doc.getElementsByTagName("width")[0].firstChild.nodeValue
    xmin = doc.getElementsByTagName("xmin")[0].firstChild.nodeValue
    ymin = doc.getElementsByTagName("ymin")[0].firstChild.nodeValue
    xmax = doc.getElementsByTagName("xmax")[0].firstChild.nodeValue
    ymax = doc.getElementsByTagName("ymax")[0].firstChild.nodeValue

    #xcent = 
    #ycent = 
