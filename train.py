from ultralytics import YOLO
import os
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np

device = 'cuda:0'
os.environ['WANDB_MODE'] = 'disabled'


def train():
    model = YOLO("yolov8n.pt")
    if __name__ == '__main__':
        results = model.train(data = "yoloConfig.yaml", epochs = 40, device = device, seed = 0)



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
    print("Agerage condifence for every prediction: ", np.mean(sum(confidencecs, [])))
