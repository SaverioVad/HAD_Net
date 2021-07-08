from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from utils.constants import *

# Data generator class
class data_generator_pretraining(Dataset):

    def __init__(self, list_ids, root_path, callee, transform=None, data_aug=False, semantics=None):
        
        # Store important information.
        self.list_ids = list_ids
        self.root_path = root_path
        self.data_aug = data_aug
        self.transform = transform
        self.semantics = semantics
        self.callee = callee 
                
    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        
        ##########################################
        ################## SETUP #################
        ##########################################

        # Extract semantic info.
        set_x, set_y, set_z = self.semantics[0]
        num_mods = self.semantics[1]
        
        # Data folder path.
        root_path = self.root_path
    
        # Get folder path for this sample.
        sample = (self.list_ids)[idx]
        sample_path = os.path.join(root_path, sample)
        
        # Get the list of modalities/sequences.
        H_or_L = os.listdir(sample_path)
        for hl in H_or_L:
            if "HGG" in hl:
                full_path = os.path.join(sample_path, "HGG")
            elif "LGG" in hl:
                full_path = os.path.join(sample_path, "LGG")
            else:
                raise Exception("ERROR: Empty folder.")
        
        # Get and sort modalities/sequences.
        modalities = os.listdir(full_path)
        modalities = sorted(modalities)

        # Data dictionary, holding the modalities/sequences, the ground truth segmentation map, and the brain mask.
        sample_dict = {
            const_MODS: None,
            const_SEG: None,
            const_MASK: None
                    }
        
        # Holds the modalities/sequences.
        x_mods = []
        
        ##########################################
        ############### BRAIN MASK ###############
        ##########################################
        
        # Get the brain mask.
        mask_path = os.path.join(full_path, "mask.npy")
        mask = np.load(mask_path)
        unpadded_mask = np.copy(mask)
        
        # Determine the required padding.
        x_init,y_init,z_init = np.shape(mask)
        
        x_diff = set_x - x_init
        y_diff = set_y - y_init
        z_diff = set_z - z_init

        x_start = x_diff//2
        x_end = x_diff - x_start
        y_start = y_diff//2
        y_end = y_diff - y_start
        z_start = z_diff//2
        z_end = z_diff - z_start

        # Pad the brain mask.
        mask = np.pad(mask,((x_start,x_end),(y_start,y_end),(z_start,z_end)))
        x_fin, y_fin, z_fin = np.shape(mask)
        if ((x_fin != set_x) or (y_fin != set_y) or (z_fin != set_z)):
            raise Exception("Error: Wrong size.")

        # Convert brain mask to 16-bit int and put it in the dictionary.       
        mask = np.int16(mask)
        sample_dict[const_MASK]= np.copy(mask)
        
        ##########################################
        ################# SEG MAP ################
        ##########################################
    
        # Get the segmentation map.
        seg_path = os.path.join(full_path, "seg.npy")
        seg = np.load(seg_path)
        seg[np.where(seg==4)] = 3
        
        # Pad the ground truth segmentation map.
        seg = np.pad(seg,((x_start,x_end),(y_start,y_end),(z_start,z_end)))
        x_fin, y_fin, z_fin = np.shape(seg)
        if ((x_fin != set_x) or (y_fin != set_y) or (z_fin != set_z)):
            raise Exception("Error: Wrong size.")

        # Convert the segmentation map to a 16-bit int and put it in the dictionary.
        seg = np.int16(seg)
        seg = np.expand_dims(seg,axis=0)
        sample_dict[const_SEG] = np.copy(seg)
        
        ##########################################
        ################### MODS #################
        ##########################################
            
        # Each folder contains 4 modalities/sequences, the brain mask, and the segmentation ground truth.
        for modality_name in modalities:

            # We only want the modalities/sequences (i.e., not the brain mask or the segmentation map).
            if ".npy" in modality_name:
                if (("mask" not in modality_name) and ("seg" not in modality_name)):
                    
                    # Get modality.
                    mod_path = os.path.join(full_path, modality_name)
                    modality = np.load(mod_path)
                    modality = np.float32(modality)
  
                    # Normalize the modalities/sequences so that they have 0 mean and unit standard deviation.
                    brain_mask = np.where(unpadded_mask==1)
                    mu = modality[brain_mask].mean()
                    sigma = modality[brain_mask].std()
                    
                    modality = (modality - mu) / sigma 
                    modality = np.clip(modality, np.min(modality),3)
                    modality = (modality + (-np.min(modality))) / (3-np.min(modality))
                    
                    # Pad the modality/sequence.
                    modality = np.pad(modality,((x_start,x_end),(y_start,y_end),(z_start,z_end)), 'constant', constant_values=(0))
                    x_fin, y_fin, z_fin = np.shape(modality)
                    if ((x_fin != set_x) or (y_fin != set_y) or (z_fin != set_z)):
                        raise Exception("Error: Wrong size.")
                    
                    # If the callee is the student, then only add the the pre-contrast modalities/sequences to the list.
                    # If it is the teacher, append all the available modalities/sequences to the list.
                    if(self.callee == "student"):
                        if("t1c" not in modality_name):
                            x_mods.append(modality)
                    elif(self.callee == "teacher"):
                        x_mods.append(modality)
                    else:
                        raise Exception("ERROR: callee type ''",self.callee,"'' not supported.")

        # Check length of the list of modalities/sequences.
        if(len(x_mods)!=num_mods):
            raise Exception("ERROR: length x_mod is not ",num_mods,"! It is of length ",len(x_mods))    
                
        # Concatenate the input modalities/sequences.
        mod_shape_0, mod_shape_1, mod_shape_2 = x_mods[0].shape
        concated_x_mods = np.zeros((num_mods, mod_shape_0, mod_shape_1, mod_shape_2))
        for mod_index in range(num_mods):
            concated_x_mods[mod_index,:,:,:] = np.copy(x_mods[mod_index])
        sample_dict[const_MODS] = np.copy(concated_x_mods)
        
        ##########################################
        ################ DATA AUG ################
        ##########################################
        
        # If true, augment and convert to tensor.
        if(self.data_aug):
            
            # Check for error.
            if(self.transform is None):
                raise Exception("ERROR: Transform is None while data_aug is True!")
            
            # Augment data.
            sample_dict = self.transform(sample_dict)
            
            # Convert to tensor.
            for key in sample_dict:
                sample_dict[key] = torch.from_numpy(sample_dict[key])
        
        # Else just convert to tensor.
        else:
            for key in sample_dict:
                sample_dict[key] = torch.from_numpy(sample_dict[key]) 
        
        # Return the input, target, and brain mask.
        return(sample_dict[const_MODS], sample_dict[const_SEG][0], sample_dict[const_MASK])