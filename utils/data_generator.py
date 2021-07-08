from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

# Data generator class
class data_generator(Dataset):

    def __init__(self, list_ids, root_path, semantics):
        
        # Store important information.
        self.list_ids = list_ids
        self.root_path = root_path
        self.semantics = semantics
                
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

        # Holds the modalities/sequences.
        x_teacher = []
        x_student = []
        
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
            raise Exception("Error: Wrong size after padding the brain mask.")
        
        # Convert brain mask to tensor.
        mask = torch.from_numpy(np.int16(mask))
        
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
            raise Exception("Error: Wrong size after padding the segmentation map.")  
        
        # Convert the segmentation map to tensor.
        y = torch.from_numpy(np.int16(seg))

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
                        raise Exception("Error: Wrong size after padding the modality.")        
        
                    # Save the modality/sequence as a tensor.
                    modality = torch.from_numpy(modality)
                    modality = modality.unsqueeze(0)
                    
                    # Append the modality/sequence to the list of teacher modalities/sequences.
                    x_teacher.append(modality)
                    
                    # Append the modality/sequence to the list of student modalities/sequences, if it is not post-contrast.
                    if ("t1c" not in modality_name):
                        x_student.append(modality)

        # Check lengths of the modality/sequence lists.
        if(len(x_teacher)!=num_teacher_mods):
            raise Exception("ERROR: length of x_teacher is not", num_teacher_mods,"! It is ", len(x_teacher),"!")
        if(len(x_student)!=num_student_mods):
            raise Exception("ERROR: length of x_student is not", num_student_mods,"! It is ", len(x_student),"!")
                    
        # Concatenate the input modalities.
        x_cat_teacher = torch.cat((x_teacher[0],x_teacher[1],x_teacher[2],x_teacher[3]), dim=0)
        x_cat_student = torch.cat((x_student[0],x_student[1],x_student[2]), dim=0)
        
        # Return the inputs and target.
        return(x_cat_teacher, x_cat_student, y, sample)