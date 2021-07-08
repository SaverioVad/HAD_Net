import torch

# Multi-Class Dice
def multi_class_dice(y_pred, y_true):
    
    # Note: y_pred has already been put through the argmax layer. 
    # y_true is the ground truth.
    
    # Three possible classes: 1,2,3. Class 0 is the background
    # Label 1 = necrotic and non-enhancing tumor core (NCR/NET)
    # Label 2 = peritumoral edema (ED)
    # Label 3 = GD-enhancing tumor (ET)
    # Note label 4 was changed to 3.

    # Flatten the image.
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # NET
    y_pred_NET = y_pred.clone()
    y_pred_NET[torch.where(y_pred_NET==1)] = -1
    y_pred_NET = torch.lt(y_pred_NET,0)

    y_true_NET = y_true.clone()
    y_true_NET[torch.where(y_true_NET==1)] = -1
    y_true_NET = torch.lt(y_true_NET,0)

    intersection_NET = torch.sum(y_pred_NET*y_true_NET)
    union_NET = torch.sum(y_pred_NET) + torch.sum(y_true_NET)
    
    dice_NET = (2.0*intersection_NET + 1.0)/(union_NET + 1.0)

    # ED
    y_pred_ED = y_pred.clone()
    y_pred_ED[torch.where(y_pred_ED==2)] = -1
    y_pred_ED = torch.lt(y_pred_ED,0)

    y_true_ED = y_true.clone()
    y_true_ED[torch.where(y_true_ED==2)] = -1
    y_true_ED = torch.lt(y_true_ED,0)

    intersection_ED = torch.sum(y_pred_ED*y_true_ED)
    union_ED = torch.sum(y_pred_ED) + torch.sum(y_true_ED)

    dice_ED = (2.0*intersection_ED + 1.0)/(union_ED + 1.0)

    # ET
    y_pred_ET = y_pred.clone()
    y_pred_ET[torch.where(y_pred_ET==3)] = -1
    y_pred_ET = torch.lt(y_pred_ET,0)

    y_true_ET = y_true.clone()
    y_true_ET[torch.where(y_true_ET==3)] = -1
    y_true_ET = torch.lt(y_true_ET,0)

    intersection_ET = torch.sum(y_pred_ET*y_true_ET)
    union_ET = torch.sum(y_pred_ET) + torch.sum(y_true_ET)

    dice_ET = (2.0*intersection_ET + 1.0)/(union_ET + 1.0)
    
    # WT
    y_pred_WT = y_pred.clone()
    y_pred_WT[torch.where(y_pred_WT!=0)] = -1
    y_pred_WT = torch.lt(y_pred_WT,0)

    y_true_WT = y_true.clone()
    y_true_WT[torch.where(y_true_WT!=0)] = -1
    y_true_WT = torch.lt(y_true_WT,0)

    intersection_WT = torch.sum(y_pred_WT*y_true_WT)
    union_WT = torch.sum(y_pred_WT) + torch.sum(y_true_WT)

    dice_WT = (2.0*intersection_WT + 1.0)/(union_WT + 1.0)
    
    # TC
    y_pred_TC = y_pred.clone()
    y_pred_TC[torch.where(y_pred_TC==1)] = -1
    y_pred_TC[torch.where(y_pred_TC==3)] = -1
    y_pred_TC = torch.lt(y_pred_TC,0)

    y_true_TC = y_true.clone()
    y_true_TC[torch.where(y_true_TC==1)] = -1
    y_true_TC[torch.where(y_true_TC==3)] = -1
    y_true_TC = torch.lt(y_true_TC,0)

    intersection_TC = torch.sum(y_pred_TC*y_true_TC)
    union_TC = torch.sum(y_pred_TC) + torch.sum(y_true_TC)

    dice_TC = (2.0*intersection_TC + 1.0)/(union_TC + 1.0)

    return dice_NET, dice_ED, dice_ET, dice_WT, dice_TC