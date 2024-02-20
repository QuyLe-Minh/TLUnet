from metrics.metrics import *
from utils import manual_crop, concat
from architecture.CNN3D import CNN3D
from postprocessing.postprocessing import Postprocessing
import torch.nn as nn

def eval(config, dataloader, model_state_dict):
    model = CNN3D().to(config.device)
    model.load_state_dict(torch.load(model_state_dict))
    print("Successful loading model!!!")

    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm3d):
    #          m.track_running_stats=False
    
    size = len(dataloader.dataset)
    dice_score_liver, iou_score_liver = 0, 0
    postprocess = Postprocessing()

    idx = 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(config.device)
            
            X_cropped_collection = manual_crop(X)
            y_cropped_collection = []
            for i in range(len(X_cropped_collection)):
                y_cropped = model(X_cropped_collection[i])[0]
                y_cropped_collection.append(y_cropped.detach().cpu())
            
            pred = torch.round(concat(y, y_cropped_collection)).to(torch.uint8)

            s = config.val[idx]
            s = s.split('/')[-1]
            torch.save(pred, "results/" + s)
            idx+=1
            # pred = postprocess(pred)
            
            dice_score_liver += dice(pred, y)
            iou_score_liver += iou(pred, y)
            
            print(dice(pred, y), s)

    dice_score_liver /= size
    iou_score_liver /= size
    print(f"Evaluation: \n Dice score liver: {(100 * dice_score_liver):>0.3f}%, IoU score liver: {(iou_score_liver * 100):>0.3f}% \n")

