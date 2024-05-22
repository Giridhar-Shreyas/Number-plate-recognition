from xml.dom.minidom import parse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_pose(image, lables):
    with open(lables) as fp:
        data = [list(map(float, line.strip().split(' '))) for line in fp]
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
            draw_pose(path_img+'/'+img, path_label+'/'+label)
            i+=1
        k+=1



def convert_Yolo8(filePath):
    with open(filePath, 'r') as file:
        file_content = file.read()
    doc = parse(filePath)
    print(doc.getElementsByTagName("")[0].firstChild.nodeValue)
    

