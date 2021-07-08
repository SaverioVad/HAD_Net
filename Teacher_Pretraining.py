# Import requires libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import random
import nibabel as nib
from monai import transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# Import .py files
from utils import constants
from utils.data_generator_pretraining import data_generator_pretraining
from models.generator_model import Generator as Model
from models.generator_model import init_weights
from utils.dice import multi_class_dice
from utils.transforms import Transform

# Parse input arguments.

parser = argparse.ArgumentParser(description='Teacher-PreTraining')

parser.add_argument('--GPU_index', type=str, default='0',
                    help='GPU to use. Default: 0')
parser.add_argument('--training_root_path', type=str, default='./BraTS_training',
                    help='Path to pre-processed training data. Default: ./BraTS_training')
parser.add_argument('--validation_root_path', type=str, default='./BraTS_training',
                    help='Path to pre-processed (local) validation data. Default: ./BraTS_training')
parser.add_argument('--testing_set', default=False, action='store_true',
                    help='Is there a local "held-out" testing set? Note this set must include ground truths (i.e., it is not an inference test). Default: False')
parser.add_argument('--testing_root_path', type=str, default='./BraTS_testing',
                    help='Path to pre-processed testing data (with ground truths). Default: ./BraTS_testing')
parser.add_argument('--training_ids_path', type=str, default='./training_ids.npy',
                    help='Path to list of training ids. Default: ./training_ids.npy')
parser.add_argument('--validation_ids_path', type=str, default='./validation_ids.npy',
                    help='Path to list of validation ids. Default: ./validation_ids.npy')
parser.add_argument('--testing_ids_path', type=str, default='./testing_ids.npy',
                    help='Path to list of testing ids. Default: ./testing_ids.npy')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size used during training/validation/testing. Default: 1')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train for. Default: 400')
parser.add_argument('--init_num_filters', type=int, default=32,
                    help='Initial number of filters. Default: 32')
parser.add_argument('--dropout_prob', type=float, default=0.2,
                    help='Dropout probability. Default: 0.2')
parser.add_argument('--num_tr_workers', type=int, default=11,
                    help='Number of workers for the training set. Default: 11')
parser.add_argument('--num_val_workers', type=int, default=11,
                    help='Number of workers for the validation set. Default: 11')
parser.add_argument('--vm_decay', type=float, default=0.95,
                    help='Value at which the running (and corrected) validation metric(s) is(are) decayed. Default: 0.95')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='Learning rate of the teacher model. Default: 0.0002')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='Value of beta_1 in the AdamW optimizer. Default: 0.9')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='Value of beta_2 in the AdamW optimizer. Default: 0.999')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Value of eps in the AdamW optimizer. Default: 1e-8')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Value of weight decay in the AdamW optimizer. Default: 1e-5')
parser.add_argument('--weight_decay_factor', type=float, default=0.98,
                    help='Factor by which the CE weights are decayed. Default: 0.98')
parser.add_argument('--augmentation_flipprob', type=float, default=0.50,
                    help='Probability of flipping input modalities. Default: 0.50')
parser.add_argument('--augmentation_affineprob', type=float, default=0.80,
                    help='Probability of applying an affine transformation to the input modalities. Default: 0.80')
parser.add_argument('--lr_scheduler_patience', type=int, default=30,
                    help='Number of epochs with no improvement after which learning rate will be reduced. Default: 30')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                    help='Factor by which the learning rate will be reduced.. Default: 0.5')
parser.add_argument('--continuous_saving', default=False, action='store_true',
                    help='Save the teacher model after each epoch? Default: False')
parser.add_argument('--save_state_dir', type=str, default="./",
                    help='Directory where the model states will be saved. Default: ./')
args = parser.parse_args()


# Get data ids and run some checks.
def init_dataset():
    
    # Load Tranining/Validation/Testing ids
    training_ids = np.load(args.training_ids_path)
    validation_ids = np.load(args.validation_ids_path)
    if(args.testing_set):
        testing_ids = np.load(args.testing_ids_path)
    else:
        testing_ids = None

    # Print dataset details.
    print("Number of training samples:", len(training_ids))
    print("Number of validation samples:", len(validation_ids))
    if(args.testing_set):
        print("Number of testing samples:", len(testing_ids))

    # Check for a data leak between the sets.
    print("Checking for data leak...")

    for tr_id in training_ids:
        for val_id in validation_ids:
            if(tr_id == val_id):
                raise Exception("ERROR: Data leak detected between the Training and Validation sets.")

    if(args.testing_set):
        for tr_id in training_ids:
            for test_id in testing_ids:
                if(tr_id == test_id):
                    raise Exception("ERROR: Data leak detected between the Training and Testing sets.")

        for val_id in validation_ids:
            for test_id in testing_ids:
                if(val_id == test_id):
                    raise Exception("ERROR: Data leak detected between the Validation and Testing sets.")

    print("No data leak detected.")
    
    return training_ids, validation_ids, testing_ids

    
# GPU selection. Raises an error if the specified GPU is not found.
def select_GPU():
    device = torch.device("cuda:"+args.GPU_index)
    try:
        torch.cuda.get_device_name(device)
    except:
        raise Exception("ERROR: GPU X not found.")
    print("Now using GPU"+args.GPU_index+".") 
    return device


# Determine maximum size of all three dimensions
def data_semantic_info():

    print("Finding maximum size of all samples in training/validation set...")

    list_x = []
    list_y = []
    list_z = []

    list_samples = os.listdir(args.training_root_path)
    for sample in list_samples:

        sample_path = os.path.join(args.training_root_path, sample)
        sample_files = os.listdir(sample_path)

        LGG = False
        HGG = False

        if "LGG" in sample_files:
            LGG_or_HGG_path = os.path.join(sample_path, "LGG")
            LGG = True
        if "HGG" in sample_files:
            LGG_or_HGG_path = os.path.join(sample_path, "HGG")
            HGG = True

        if((LGG == True) and (HGG == True)):
            raise Exception("Error: Both LGG and HGG are true.")

        if((LGG == False) and (HGG == False)):
            raise Exception("Error: Neither LGG or HGG are true.")

        list_mods = os.listdir(LGG_or_HGG_path)
        
        try:
            mod_path = os.path.join(LGG_or_HGG_path, "seg.npy")
            modality = np.load(mod_path)
            modality = np.float32(modality)

            x,y,z = np.shape(modality)

            list_x.append(x)
            list_y.append(y)
            list_z.append(z)

        except Exception:
            raise Exception("Error: Could not ground truth for sample",sample)

    maximum_x_val = np.max(list_x)
    maximum_y_val = np.max(list_y)
    maximum_z_val = np.max(list_z)

    print("Maximum size:")
    print("dim0 =",maximum_x_val)
    print("dim1 =",maximum_y_val)
    print("dim2 =",maximum_z_val)
    print(" ")

    # Find padding values.
    set_x = int(np.ceil(maximum_x_val/16.0)*16.0)
    set_y = int(np.ceil(maximum_y_val/16.0)*16.0)
    set_z = int(np.ceil(maximum_z_val/16.0)*16.0)
    set_x = np.maximum(set_x,set_z)
    set_z = set_x

    print("Post-padding size will be:")
    print("dim0 =",set_x)
    print("dim1 =",set_y)
    print("dim2 =",set_z)
    print(" ")

    # For the BraTS dataset, the number of teacher modalities/sequences is 4.
    num_mods_teacher = 4
    
    # Create semantic info.
    semantics = ((set_x, set_y, set_z), num_mods_teacher)
    return semantics


# Get transforms for data augmentation. 
def get_transform():
      
    # Hyperparameters
    augmentation_rotate = 3*(4*np.pi/180,)
    augmentation_shear = 3*(0.04,)
    augmentation_scale = 3*(0.04,)
      
    # Create instance of data augmentation class.
    transform = Transform(args.augmentation_flipprob, args.augmentation_affineprob, augmentation_rotate, augmentation_shear, augmentation_scale)
    
    return transform


def get_dataloaders(list_ids, root_paths, semantics, transform):

    # Create training dataloader.
    data_gen_train = data_generator_pretraining(list_ids[0], root_paths[0], "teacher", transform, True, semantics)
    trainset = torch.utils.data.DataLoader(data_gen_train, batch_size = args.batch_size, shuffle = True, num_workers=args.num_tr_workers)
    print("Created training dataloader.")

    # Create validation dataloader.
    data_gen_val = data_generator_pretraining(list_ids[1], root_paths[1], "teacher", None, False, semantics)
    valset = torch.utils.data.DataLoader(data_gen_val, batch_size = args.batch_size, shuffle = True, num_workers=args.num_val_workers)
    print("Created validation dataloader.")

    # Create testing dataloader.
    if(args.testing_set):
        data_gen_test = data_generator_pretraining(list_ids[2], root_paths[2], "teacher", None, False, semantics)
        testset = torch.utils.data.DataLoader(data_gen_test, batch_size = args.batch_size, shuffle = False)
        print("Created testing dataloader.")
        
        return trainset, valset, testset
        
    return trainset, valset, None


# Create instance of the model and put it on the GPU.
def get_model(device):
    
    # Teacher network. Note: for BraTS, the total number of modalities/sequences is 4. Also, BraTS has 4 segmentation classes.
    model = Model(num_mods=4, k=args.init_num_filters, p=args.dropout_prob, num_class=4).to(device)
    print("Created teacher model.")
    
    # Initialize U-Net model wieghts.
    model.apply(init_weights)
    
    return model


# Initialize the optimizers.
def get_optimizer(model):

    # Select an optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta_1,args.beta_2), eps=args.eps, weight_decay=args.weight_decay)
    
    return optimizer


# Create the learning rate scheduler.
def get_lr_scheduler(optimizer):
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience, verbose=True)
    
    return scheduler


# Train for an epoch. 
def training_epoch(model, optimizer, trainset, weights, device, epoch):
    
    # Sample counter 
    sample_count = 0
    
    # Loss tracker
    sum_loss = 0
    
    # Update CE wieghts
    if(epoch>0):
        weights = weights*args.weight_decay_factor
        weights = torch.max(weights, torch.ones(len(weights)))
        loss_function = nn.CrossEntropyLoss(weight = weights).to(device) 
    else:
        loss_function = nn.CrossEntropyLoss(weight = weights).to(device)
            
    # Set model to train mode.
    model.train()
    
    # Go over each batch of the training set
    for data in trainset:
        
        # Verbose
        if(sample_count%50 ==0):
            print("Sample ",sample_count,", Epoch",epoch)
            
        # Extract input and target
        x = data[0].float().to(device)
        y = data[1].long().to(device)
        
        # Run the batch through the CNN
        output, _ = model(x)
    
        # Compute loss.
        loss = loss_function(output, y)
        
        # zero the gradient
        optimizer.zero_grad()
        
        # Step forward.
        loss.backward()
        optimizer.step()
        
        # Move data from GPU to CPU.
        x = x.detach().to('cpu')
        y = y.detach().to('cpu')
        loss = loss.detach().to('cpu')
        output = output.detach().to('cpu')
        
        # Sum loss for this epoch.
        sum_loss = sum_loss + loss.item()

        # Increment counter. 
        sample_count+=1.0        
    
    # Get epoch loss.
    mean_loss = sum_loss/sample_count
    
    return model, optimizer, weights, loss_function, mean_loss
    
    
# Perform validation for this epoch.
def validation_epoch(model, loss_function, valset, n_softmax, running_vm, validate_vm_list, device, epoch):
    
    # Store validation metrics.
    dice_scores_NET = 0.0
    dice_scores_ED = 0.0
    dice_scores_ET = 0.0
    dice_scores_WT = 0.0
    dice_scores_TC = 0.0
    validation_loss = 0.0
    
    # Counter
    sample_count = 0.0
    
    # Set to eval.
    model.eval()
   
    # Go over each batch of the validation set
    for data in valset:
        
        # Verbose
        if(sample_count%50 ==0):
            print("Sample ",sample_count,", Epoch",epoch)
    
        # Extract input and target
        x = data[0].float().to(device)
        y = data[1].long().to(device)

        # No gradients.
        with torch.no_grad():
            
            # Run the batch through the CNN
            output, _ = model(x)

            # find loss
            val_loss = loss_function(output, y)
        
        # Append loss to list.
        validation_loss = validation_loss + val_loss.detach().to('cpu').item()
        
        # Get the output image.
        output_image = n_softmax(output)
        output_image = torch.argmax(output_image, dim = 1)

        # Compute Dice score
        dice_NET, dice_ED, dice_ET, dice_WT, dice_TC = multi_class_dice(output_image, y)
        dice_scores_NET = dice_scores_NET + dice_NET.item()
        dice_scores_ED = dice_scores_ED + dice_ED.item()
        dice_scores_ET = dice_scores_ET + dice_ET.item() 
        dice_scores_WT = dice_scores_WT + dice_WT.item()
        dice_scores_TC = dice_scores_TC + dice_TC.item()
        
        # Increment counter
        sample_count+=1.0

    # Mean val loss.
    mean_val_loss = validation_loss/sample_count
    
    # Compute mean dice scores
    mean_dice_NET = dice_scores_NET/sample_count
    mean_dice_ED = dice_scores_ED/sample_count
    mean_dice_ET = dice_scores_ET/sample_count
    mean_dice_WT = dice_scores_WT/sample_count
    mean_dice_TC = dice_scores_TC/sample_count

    # Print mean dice scores for this epoch
    print("mean_NET =", mean_dice_NET)
    print("mean_ED =", mean_dice_ED)
    print("mean_ET =", mean_dice_ET)
    print("mean_WT =", mean_dice_WT)
    print("mean_TC =", mean_dice_TC)  
    
    # Validation Metrics
    vm_weights = [1.0,1.0,1.0]
    vm_weights = vm_weights/(np.sum(vm_weights))
    
    # Compute validation metric
    validate_vm = vm_weights[0]*mean_dice_WT + vm_weights[1]*mean_dice_TC + vm_weights[2]*mean_dice_ET
    
    # Compute running metric
    running_vm = (args.vm_decay)*running_vm + (1.0-args.vm_decay)*validate_vm
    
    # Compute corrected metric
    corrected_vm = running_vm/(1-args.vm_decay**(epoch+1))

    # Append to list of mean dice scores.
    validate_vm_list.append(validate_vm)
              
    return mean_val_loss, (mean_dice_NET,mean_dice_ED,mean_dice_ET,mean_dice_WT,mean_dice_TC), (validate_vm_list, running_vm, corrected_vm)



def main():
        
    ###########################################
    ################## INITS ##################
    ###########################################
              
    # Get data ids and run some checks.
    training_ids, validation_ids, testing_ids = init_dataset()
              
    # GPU selection. Raises an error if the specified GPU is not found.
    device = select_GPU()
    
    # Determine maximum size of all three dimensions.
    semantics = data_semantic_info()
    
    # Get transforms for data augmentation. 
    transform = get_transform()
    
    # Get dataloaders for the training, validation, and testing sets.
    trainset, valset, testset = get_dataloaders((training_ids, validation_ids, testing_ids),(args.training_root_path, args.validation_root_path, args.testing_root_path), semantics, transform)

    # Create instance of the models and put it on the GPU
    model = get_model(device)
    
    # Make softmax layer.
    n_softmax = nn.Softmax(dim=1).to(device)

    # Initialize the optimizer.
    optimizer = get_optimizer(model)
    
    # Get the lr_scheduler.
    scheduler = get_lr_scheduler(optimizer)
    
    # Loss function weights. 
    weights = torch.tensor([1.0000, 59.4507, 23.3190, 71.8481]) # specifically found for the BraTS 2019 dataset
              
    ###########################################
    ######### TRAINING AND VALIDATION #########
    ###########################################
              
    # List of validation metrics.
    validate_vm_list = []

    # Running validation metirc 
    running_vm = 0.0

    # For each Epoch
    for epoch in range(args.epochs):

        # TRAINING
        print(" ")
        print("Beginning training for epoch",epoch,".")
        print(" ")

        # Train for one epoch.
        model, optimizer, weights, loss_function, mean_loss = training_epoch(model, optimizer, trainset, weights, device, epoch)
               
        # VALIDATION
        print(" ")
        print("Beginning validation for epoch",epoch,".")
        print(" ")
              
        # Validation for this epoch.
        mean_val_loss, mean_dice_scores, vm_info = validation_epoch(model, loss_function, valset, n_softmax, running_vm, validate_vm_list, device, epoch)
              
        mean_dice_NET, mean_dice_ED, mean_dice_ET, mean_dice_WT, mean_dice_TC = mean_dice_scores
        validate_vm_list, running_vm, corrected_vm = vm_info
    
        ###########################################
        ########## SAVE MODEL PARAMETERS ##########
        ###########################################

        # Get initial dir.
        original_dir = os.getcwd()
        os.chdir(args.save_state_dir)

        # Save best model.
        if(validate_vm_list[-1] >= np.max(validate_vm_list)):
            torch.save(model.state_dict(), "Pre_Trained_Teacher_BEST_VAL_VM.pth")
            print("Model parameters saved: best VALIDATE_VM.")

        # Save model, optimizer, and scheduler for the current epoch.
        if(args.continuous_saving):
            torch.save(model.state_dict(),"Pre_Trained_Teacher_Model.pth")
            torch.save(optimizer.state_dict(), "Pre_Trained_Teacher_Optimizer.pth") 
            torch.save(scheduler.state_dict(), "Pre_Trained_Teacher_Scheduler.pth")
            print("State dictionaries saved for this epoch.")

        # Go back to original dir.
        os.chdir(original_dir)

        # Epoch complete. 
        print("Epoch", epoch, "complete.")
        print("------------------------------------------------------------------------")

    ###########################################
    ################# TESTING #################
    ###########################################
     
    # Verbose
    print("Running best model on the local testing set.")
              
    # Load best model
    original_dir = os.getcwd()
    os.chdir(args.save_state_dir)
    model.load_state_dict(torch.load("Pre_Trained_Teacher_BEST_VAL_VM.pth"))
    os.chdir(original_dir)
              
    # Run model on testing set.
    mean_test_loss, mean_dice_scores, vm_info = validation_epoch(model, loss_function, testset, n_softmax, 0.0, [], device, 0)

    mean_dice_NET, mean_dice_ED, mean_dice_ET, mean_dice_WT, mean_dice_TC = mean_dice_scores
    testing_vm_list, running_vm, corrected_vm = vm_info
    
    # Print results.
    print(" ")
    print("Testing Mean NET Dice:", mean_dice_NET)
    print("Testing Mean ED Dice:", mean_dice_ED)
    print("Testing Mean ET Dice:", mean_dice_ET)
    print("Testing Mean WT Dice:", mean_dice_WT) 
    print("Testing Mean TC Dice:", mean_dice_TC)
    print("Testing Mean Loss:", mean_test_loss)
    
if __name__ == '__main__':
    main()