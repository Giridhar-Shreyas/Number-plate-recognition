from ultralytics import YOLO
import os
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
import utils as u

device = 'cuda:0'
os.environ['WANDB_MODE'] = 'disabled'



def train():
    model = YOLO("yolov8n.pt")
    if __name__ == '__main__':
        results = model.train(data = "yoloConfig.yaml", epochs = 200, device = device, seed = 0, patience=20)



def test(best_weights):
    #best_weights = "runs\detect\\train14\weights\\best.pt"
    path_img =  os.getcwd() + "\\data\\final_data\\test\images\\"    
    path_label = os.getcwd() + "\\data\\final_data\\test\labels\\"


    model = YOLO(best_weights)

    
    losses = []
    confidencecs = []
    results = []
    r_i = 0
    loop = 0

    for img, label in zip(os.listdir(path_img), os.listdir(path_label)):
        result = model(source = Image.open(path_img+img))
        with open(path_label+label) as fp:
            data = [list(map(float, line.strip().split(' '))) for line in fp]
            data = torch.tensor(np.array(data), device=device, dtype=torch.float32)

        
        for r in result:
            pred_poses = r.boxes.xywhn
            results.append([])
            confidencecs.append([])
            losses.append([])
            num_prediction = len(data)
            ### ugly fix for if the prediceted poses contain some nonsense in the background ###
        if num_prediction == len(pred_poses):
            for i in range(len(pred_poses)):
                print(F.mse_loss(pred_poses[i], data[i][1:]).to('cpu').item())
                losses[r_i].append(F.mse_loss(pred_poses[i], data[i][1:]).to('cpu').item())
                confidencecs[r_i].append(r.boxes.conf[i].to('cpu').item())
                results[r_i].append(pred_poses[i].to('cpu'))
        ### just check the one with least mse error ###
        else:
            for i in range(len(data)):
                best_loss =  float('inf')
                best_confidence = 0.0
                best_result = torch.ones_like(pred_poses, device=device)
                for j in range(len(pred_poses)):
                    loss = F.mse_loss(pred_poses[j], data[i][1:]).to('cpu').item()
                    print(loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_confidence = r.boxes.conf[j].to('cpu').item()
                        best_result = pred_poses[j].to('cpu')
                losses[r_i].append(best_loss)
                confidencecs[r_i].append(best_confidence)
                results[r_i].append(best_result)  
        loop +=1
        r_i+=1
    print("################################################################################################")
    print("Number of images tested: ", (loop+1))
    print("Average MSE losses: ", np.mean(sum(losses, [])))
    print("Agverage condifence for every prediction: ", np.mean(sum(confidencecs, [])))


def predict(img_path, model_path, num_plate=1):
    """"
    Args:
        img_path (str): path to the image
        model_path (str): path to the weights that is going to be used to prediciton
        num_plate (int): tells how many number plates are expected in an image, default value 1
    """
    model = YOLO(model_path)
    img = Image.open(img_path)
    if img.size[0] != 640 and img.size[1] !=640:
        img = img.resize((640,640))
    prediction = model(source=img)
    for result in prediction:
        if result.boxes.conf.size()[0] > 1 and num_plate==1:
            best_index = 0
            best_conf = float('inf')
            for j in range(0,result.boxes.conf.size()[0]):
                if best_conf<result.boxes.conf.to('cpu')[j]:
                    best_index = j
                    best_conf = result.boxes.conf.to('cpu')[j]
            return result.boxes.xywhn[best_index].to('cpu')


        else:
            return result.boxes.xywhn.to('cpu')
    #u.draw_pose()
    

def predict_draw(img_path, bbox):
    """
    Args:
        bbox (tensor): tensor of pose in format : anchor x, anchor y, width, height
                       with shape (num_plate, 4)
    """

    u.draw_pose(img_path, bbox)




#bbox = predict(img_path = os.getcwd()+"\\test_data\images\\test.jpg", model_path = "runs\detect\\train14_dropout\weights\\best.pt")
#u.draw_pose(image = os.getcwd()+"\\test_data\images\\test.jpg", data = bbox, offset=False)


#bbox = predict(img_path = os.getcwd()+"\\test_data\images\\test_adjecent.jpg", model_path = "runs\detect\\train14_dropout\weights\\best.pt",num_plate=2)
#u.draw_pose(image = os.getcwd()+"\\test_data\images\\test_adjecent.jpg", data = bbox, offset=False)



#train()
#test("runs\detect\\train14_dropout\weights\\best.pt")