from metrics.metrics import *
from utils import one_hot_encoder, manual_crop, concat
from architecture.TLUnet import TLUnet

def eval(config, dataloader, model_state_dict):
    model = TLUnet().to(config.device)
    model.load_state_dict(model_state_dict)
    print("Successful loading model!!!")   
    
    size = len(dataloader.dataset)
    dice_score_liver, iou_score_liver = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y_one_hot = one_hot_encoder(y)
            
            X_cropped_collection = manual_crop(X)
            y_cropped_collection = []
            for i in range(len(X_cropped_collection)):
                y_cropped = model(X_cropped_collection[i])[0]
                y_cropped_collection.append(y_cropped.detach().cpu())
            
            pred = concat(y_one_hot, y_cropped_collection)
            pred = one_hot_encoder(torch.argmax(pred, dim = 1))
            
            dice_score_liver += dice(pred[:, 1], y_one_hot[:, 1])
            iou_score_liver += iou(pred[:, 1], y_one_hot[:, 1])

    dice_score_liver /= size
    iou_score_liver /= size
    print(f"Evaluation: \n Dice score liver: {(100 * dice_score_liver):>0.3f}%, IoU score liver: {(iou_score_liver * 100):>0.3f}% \n")
    return