from xml.dom.minidom import parse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_pose(image, data):
    """
    Args: 
        image (str): Path to the image with the resolution 640x640
        data list[[float]]: A 2-d list of exactly 5 values with format [class, x center, y center, width, height]
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
    path_img = os.getcwd() + '/data/final_data/train/images'
    path_label = os.getcwd() + '/data/final_data/train/labels'
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
        img = os.getcwd() + '/data/data_1/images/N'+str(ii)+'.jpeg'
        label = os.getcwd() + '/data/data_1/images/N'+str(ii)+'.xml'
        rezise_img(img, os.getcwd() + '/data/final_data/train/images/N'+str(ii)+'.jpeg')
        data = convert_label(label)
        if data is not None:
            with open(os.getcwd() + '/data/final_data/train/lables/N'+str(ii)+'.txt','w') as f:
                f.write(data)


def rezise_img(imgFile, saveTo):
    """
    Args:
        imgFile (str): Path to the image needed to be resized 
        saveTo (str): Path to where the image needs to be be saved
    """
    img = imgFile
    try:
        image = Image.open(img)
        image = image.resize((640,640))
        image.save(saveTo)
    except OSError as e:
        pass

    

def convert_label(label):
    """
    Args:
        label (str): File path to the label in xml format
        key tags: Height, Width, bounding box (xmin, ymin, xmax, ymax) 
    Returns:
        lables in yolo format and 
    """
    try:
        doc = parse(label)
        resHeight = float(doc.getElementsByTagName("height")[0].firstChild.nodeValue)
        resWidth = float(doc.getElementsByTagName("width")[0].firstChild.nodeValue)
        xmin = float(doc.getElementsByTagName("xmin")[0].firstChild.nodeValue)/resWidth
        ymin = float(doc.getElementsByTagName("ymin")[0].firstChild.nodeValue)/resHeight
        xmax = float(doc.getElementsByTagName("xmax")[0].firstChild.nodeValue)/resWidth
        ymax = float(doc.getElementsByTagName("ymax")[0].firstChild.nodeValue)/resHeight

        width = xmax-xmin
        height = ymax-ymin
        xcent = xmin+(width/2.0)
        ycent = ymin+(height/2.0)

        return str(0)+' '+str(xcent)+' '+str(ycent)+' '+str(width)+' '+str(height)
    except OSError as e:
        return None

test_draw_pose()