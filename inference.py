# Import requires libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import argparse

# Import .py files
from utils import constants
from utils.data_generator_inference import data_generator_inference
from models.generator_model import Generator

# Parse input arguments.
parser = argparse.ArgumentParser(description='HAD-Net')

parser.add_argument('--GPU_index', type=str, default='0',
                    help='GPU to use. Default: 0')
parser.add_argument('--type', type=str, default='student',
                    help='Model type (i.e., student model or teacher model). If "student" is selected, the model will only receive pre-contrast sequences (i.e., 3) as input; whereas, if "teacher" is selected, the model will receive all sequences (i.e., 4). Default: student')
parser.add_argument('--inference_proc_root_path', type=str, default='./BraTS_validation',
                    help='Path to pre-processed validation data (without ground truth segmentations). Default: ./BraTS_validation')
parser.add_argument('--inference_original_root_path', type=str, default='./BraTS_validation_original',
                    help='Path to the original (without pre-processing) validation data (without ground truth segmentations). Default: ./BraTS_validation_original')
parser.add_argument('--inference_ids_path', type=str, default='./inference_ids.npy',
                    help='Path to list of inference ids. Default: ./inference_ids.npy')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size used during training/validation/testing. Default: 1')
parser.add_argument('--init_num_filters', type=int, default=32,
                    help='Initial number of filters. Default: 32')
parser.add_argument('--model_path', type=str, default="./HAD_Net_BEST_VAL_VM.pth",
                    help='Path to the state dictionary of the best HAD-Net model. Default: ./HAD_Net_BEST_VAL_VM.pth')
parser.add_argument('--semantics_path', type=str, default="./HAD_Net_semantics.npy",
                    help='Path to the semantic information saved by HAD_Net.py at the start of training. Default: ./HAD_Net_semantics.npy')
parser.add_argument('--output_dir', type=str, default="./",
                    help='Directory where the output segmentations will be saved. Default: ./')
args = parser.parse_args()


# Get data ids and run some checks.
def init_dataset():
    
    # Load Tranining/Validation/Testing ids
    inference_ids = np.load(args.inference_ids_path)
    print("Number of inference samples:", len(inference_ids))

    return inference_ids

    
# GPU selection. Raises an error if the specified GPU is not found.
def select_GPU():
    device = torch.device("cuda:"+args.GPU_index)
    try:
        torch.cuda.get_device_name(device)
    except Exception:
        raise Exception("ERROR: GPU X not found.")
        
    print("Now using GPU"+args.GPU_index+".") 
    return device


# Determine maximum size of all three dimensions
def data_semantic_info():

    # Load semantic info from training.
    semantics = np.load(args.semantics_path, allow_pickle=True)
    
    return semantics


# Get the dataloader.
def get_dataloader(list_ids, root_path, semantics):

    # Create inference dataloader.
    data_gen_infer = data_generator_inference(list_ids, root_path, semantics, args.type)
    inferset = torch.utils.data.DataLoader(data_gen_infer, batch_size = args.batch_size, shuffle = False)
    print("Created inference dataloader.")
        
    return inferset


# Create instance of the model and put it on the GPU.
def get_model(device):  
    
    # Student network. Note: For BraTS, there is only 1 post-contrast modalities/sequences. 
    # Also, BraTS has 4 segmentation classes.
    if(args.type == "teacher"):
        generator_model = Generator(num_mods=4, k=args.init_num_filters, p=0.2, num_class=4).to(device)
        print("Created generator model (i.e., the teacher model).")
    elif(args.type == "student"):
        generator_model = Generator(num_mods=3, k=args.init_num_filters, p=0.2, num_class=4).to(device)
        print("Created generator model (i.e., the student model).")
    else:
        raise Exception("ERROR: Unsupported model type",args.type,".") 
    
    return generator_model


# Load model.
def load_model(generator_model):

    # LOAD TRAINED GENERATOR
    trained_statedict = torch.load(args.model_path)
    generator_model.load_state_dict(trained_statedict)
    print("Loaded best model.")
    print(" ")
    
    return generator_model


# Determine the cuts that were made during pre-processing.
def get_cuts(sample_name):
      
    # List of cuts.
    x_start_list = []
    x_end_list = []
    y_start_list = []
    y_end_list = []
    z_start_list = []
    z_end_list = []
        
    # Loop through modalities to get final padding information.
    for mod_name in ["_flair.nii.gz", "_t1ce.nii.gz", "_t1.nii.gz", "_t2.nii.gz"]:
        
        # Get original mod.
        mod = nib.load(os.path.join(args.inference_original_root_path,sample_name[0],sample_name[0]+mod_name))
        mod = np.asarray(mod.dataobj)
        
        # Determine applied padding for this mod.
        non_zero_elements = np.nonzero(mod)
        x_start_list.append(np.min(non_zero_elements[0]))
        x_end_list.append(mod.shape[0] - np.max(non_zero_elements[0]) - 1)
        y_start_list.append(np.min(non_zero_elements[1]))
        y_end_list.append(mod.shape[1] - np.max(non_zero_elements[1]) - 1)
        z_start_list.append(np.min(non_zero_elements[2]))
        z_end_list.append(mod.shape[2] - np.max(non_zero_elements[2]) - 1)
        
    return max(x_start_list), max(x_end_list), max(y_start_list), max(y_end_list), max(z_start_list), max(z_end_list)       


# Post-process and save the segmentation map, such that it can be submitted to the online Validation set evaluator.
# Note that this function saves segmentation as a ".nii.gz" file that is of the same shape as the original input images.
def save_segmentation(segmap, padding_info, sample_name):
    
        # Convert label 3 to label 4.
        segmap[torch.where(segmap==3)] = 4

        # Convert to numpy.
        segmap = segmap.detach().cpu().numpy()
        segmap = segmap[0]

        # Extract padding info.
        init_x_start = padding_info[0][0]
        init_x_end = padding_info[0][1]
        init_y_start = padding_info[1][0]
        init_y_end = padding_info[1][1]
        init_z_start = padding_info[2][0]
        init_z_end = padding_info[2][1]
        flag_x = padding_info[3][0]
        flag_y = padding_info[3][1]
        flag_z = padding_info[3][2]
        
        # Get current shape.
        init_x, init_y, init_z = segmap.shape
        
        # Cut initial padding.
        if(flag_x):
            segmap = np.pad(segmap,((init_x_start,init_x_end),(0,0),(0,0)), 'constant', constant_values=(0))
        else:
            segmap = segmap[init_x_start:init_x-init_x_end,:,:]
        if(flag_y):
            segmap = np.pad(segmap,((0,0),(init_y_start,init_y_end),(0,0)), 'constant', constant_values=(0))
        else:
            segmap = segmap[:,init_y_start:init_y-init_y_end,:]
        if(flag_z):
            segmap = np.pad(segmap,((0,0),(0,0),(init_z_start,init_z_end)), 'constant', constant_values=(0))
        else:
            segmap = segmap[:,:,init_z_start:init_z-init_z_end]
            
        # Swap and flip axes.
        segmap = np.swapaxes(segmap,0,2)
        segmap = np.flip(segmap,1)

        # Getting final padding information.
        x_start, x_end, y_start, y_end, z_start, z_end = get_cuts(sample_name)
        
        # Final padding.
        segmap = np.pad(segmap,((x_start,x_end),(y_start,y_end),(z_start,z_end)))
       
        # Get affine information.
        original_mod = nib.load(os.path.join(args.inference_original_root_path,sample_name[0],sample_name[0]+"_flair.nii.gz"))
        original_mod_affine = original_mod.affine
    
        # Check final shape.
        if(segmap.shape != original_mod.shape):
            raise Exception("ERROR: Segmentation map is of size",segmap.shape,", but it should be of size",original_mod.shape,".")
            
        # Make nib file.
        segmap_NIB = nib.Nifti1Image(segmap, original_mod_affine)

        # Save seg map.
        nib.save(segmap_NIB, os.path.join(args.output_dir, sample_name[0]+".nii.gz")) 

# Perform inference.
def inference(generator_model, inferset, n_softmax, device):

    # Set to eval.
    generator_model.eval()

    # Counter.
    count = 0

    # Go over each batch of the validation set
    for data in inferset:

        # Extract input and target
        x_mods = data[0].float().to(device)
        sample_name = data[1]
        padding_info = data[2]

        # No gradients. 
        with torch.no_grad():
            
            # Run the input data through the generator.
            segmap, features = generator_model(x_mods)

        # Get the output image.
        segmap = n_softmax(segmap)
        segmap = torch.argmax(segmap, dim=1)
            
        # Post-process and save the segmentation map as a ".nii.gz" file.
        save_segmentation(segmap, padding_info, sample_name)

        # Increment
        count+=1
        print(sample_name[0],"complete (",(int(10000*count/inferset.__len__()))/100.0,"% )")
        
        
def main():
           
    ###########################################
    ################## INITS ##################
    ###########################################
              
    # Get data ids and run some checks.
    inference_ids = init_dataset()
              
    # GPU selection. Raises an error if the specified GPU is not found.
    device = select_GPU()
    
    # Get semantic info from training.
    semantics = data_semantic_info()
    
    # Get dataloaders for the inference set.
    inferset = get_dataloader(inference_ids, args.inference_proc_root_path, semantics)

    # Create instance of the model and put it on the GPU.
    generator_model = get_model(device)
              
    # Get softmax.
    n_softmax = nn.Softmax(dim=1).to(device)

    # Load model state dict.
    generator_model = load_model(generator_model)
              
    ###########################################
    ################ INFERENCE ################
    ###########################################
              
    # Perform inference.
    inference(generator_model, inferset, n_softmax, device)
    
    print("Inference complete.")
    

if __name__ == '__main__':
    main()