from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

# Data generator class
class data_generator_inference(Dataset):

    def __init__(self, list_ids, root_path, semantics, model_type):
        
        # Store important information.
        self.list_ids = list_ids
        self.root_path = root_path
        self.semantics = semantics
        self.model_type = model_type
                
    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):

        # Extract semantic info.
        set_x, set_y, set_z = self.semantics[0]
        num_teacher_mods = self.semantics[1]
        num_student_mods = self.semantics[2]
        
        # Data folder path.
        root_path = self.root_path
    
        # Get folder path for this sample.
        sample = (self.list_ids)[idx]
        full_path = os.path.join(root_path, sample, "LGG_or_HGG")
        
        # Get and sort modalities/sequences.
        modalities = os.listdir(full_path)
        modalities = sorted(modalities)

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

        x_start = np.absolute(x_diff//2)
        x_end = np.absolute(x_diff) - x_start
        y_start = np.absolute(y_diff//2)
        y_end = np.absolute(y_diff) - y_start
        z_start = np.absolute(z_diff//2)
        z_end = np.absolute(z_diff) - z_start
        
        flag_x = False
        flag_y = False
        flag_z = False
        
        if(x_diff<0):
            print("Warning: Sample",sample,"cutoff in dim0.")
            flag_x = True
        if(y_diff<0):
            print("Warning: Sample",sample,"cutoff in dim1.")
            flag_y = True
        if(z_diff<0):
            print("Warning: Sample",sample,"cutoff in dim2.")
            flag_z = True

        # Pad the brain mask.
        if(flag_x):
            mask = mask[x_start:x_init-x_end,:,:]
        else:
            mask = np.pad(mask,((x_start,x_end),(0,0),(0,0)))
        if(flag_y):
            mask = mask[:,y_start:y_init-y_end,:]
        else:
            mask = np.pad(mask,((0,0),(y_start,y_end),(0,0)))
        if(flag_z):
            mask = mask[:,:,z_start:z_init-z_end]
        else:
            mask = np.pad(mask,((0,0),(0,0),(z_start,z_end)))
                
        # Check padding.
        x_fin, y_fin, z_fin = np.shape(mask)
        if ((x_fin != set_x) or (y_fin != set_y) or (z_fin != set_z)):
            print("Should be", (set_x,set_y,set_z),", but it is",(x_fin,y_fin,z_fin))
            raise Exception("Error: Wrong size after padding the brain mask.")
        
        # Convert brain mask to tensor.
        mask = torch.from_numpy(np.int16(mask))
        
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
                    if(flag_x):
                        modality = modality[x_start:x_init-x_end,:,:]
                    else:
                        modality = np.pad(modality,((x_start,x_end),(0,0),(0,0)), 'constant', constant_values=(0))
                    if(flag_y):
                        modality = modality[:,y_start:y_init-y_end,:]
                    else:
                        modality = np.pad(modality,((0,0),(y_start,y_end),(0,0)), 'constant', constant_values=(0))
                    if(flag_z):
                        modality = modality[:,:,z_start:z_init-z_end]
                    else:
                        modality = np.pad(modality,((0,0),(0,0),(z_start,z_end)), 'constant', constant_values=(0))
                    
                    # Check padding.
                    x_fin, y_fin, z_fin = np.shape(modality)
                    if ((x_fin != set_x) or (y_fin != set_y) or (z_fin != set_z)):
                        print("Should be", (set_x,set_y,set_z),", but it is",(x_fin,y_fin,z_fin))
                        raise Exception("Error: Wrong size after padding the modality.")        
        
                    # Save the modality/sequence as a tensor.
                    modality = torch.from_numpy(modality)
                    modality = modality.unsqueeze(0)
                                        
                    # Append the modality/sequence to the list of modalities/sequences.
                    # If this is the student network, only append pre-contrast modalities/sequecnes.
                    if(self.model_type == "teacher"):
                        x_mods.append(modality)
                    elif(self.model_type == "student"): 
                        if ("t1c" not in modality_name):
                            x_mods.append(modality)
                    else:
                        raise Exception("ERROR: Unsupported model type",self.model_type,".")

        # Check lengths of the modality/sequence list.
        if(self.model_type == "teacher"):
            if(len(x_mods)!=num_teacher_mods):
                raise Exception("ERROR: length of x_mods is not", num_teacher_mods,"! It is ", len(x_mods),"!")
        elif(self.model_type == "student"): 
            if(len(x_mods)!=num_student_mods):
                raise Exception("ERROR: length of x_mods is not", num_student_mods,"! It is ", len(x_mods),"!")
        else:
            raise Exception("ERROR: Unsupported model type",self.model_type,".")
                    
        # Concatenate the input modalities.
        if(self.model_type == "teacher"):
            x_cat_mods = torch.cat((x_mods[0],x_mods[1],x_mods[2],x_mods[3]), dim=0)
        elif(self.model_type == "student"):
            x_cat_mods = torch.cat((x_mods[0],x_mods[1],x_mods[2]), dim=0)
        else:
            raise Exception("ERROR: Unsupported model type",self.model_type,".")            
        
        # Padding information for post-processing.
        padding_info = ((x_start, x_end),(y_start, y_end),(z_start, z_end), (flag_x, flag_y, flag_z))
        
        # Return the inputs and other info.
        return (x_cat_mods, sample, padding_info)