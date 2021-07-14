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
from utils.data_generator import data_generator
from models.generator_model import Generator
from models.discriminator_model import Discriminator
from utils.dice import multi_class_dice

# Parse input arguments.
parser = argparse.ArgumentParser(description='HAD-Net')

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
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train for. Default: 800')
parser.add_argument('--init_num_filters', type=int, default=32,
                    help='Initial number of filters. Default: 32')
parser.add_argument('--dropout_prob', type=float, default=0.2,
                    help='Dropout probability. Default: 0.2')
parser.add_argument('--num_tr_workers', type=int, default=11,
                    help='Number of workers for the training set. Default: 11')
parser.add_argument('--num_val_workers', type=int, default=11,
                    help='Number of workers for the validation set. Default: 11')
parser.add_argument('--lambda_const', type=float, default=0.2,
                    help='Value of lambda (for student network loss). Default: 0.2')
parser.add_argument('--disc_threshold', type=float, default=0.8,
                    help='Value of the discriminator threshold. Default: 0.8')
parser.add_argument('--vm_decay', type=float, default=0.95,
                    help='Value at which the running (and corrected) validation metric(s) is(are) decayed. Default: 0.95')
parser.add_argument('--gen_lr', type=float, default=0.0002,
                    help='Learning rate of generator. Default: 0.0002')
parser.add_argument('--disc_lr', type=float, default=0.0002,
                    help='Learning rate of discriminator. Default: 0.0002')
parser.add_argument('--beta_1', type=float, default=0.5,
                    help='Value of beta_1 in the Adam optimizer (used for both generator and discriminator). Default: 0.5')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='Value of beta_2 in the Adam optimizer (used for both generator and discriminator). Default: 0.999')
parser.add_argument('--pre_trained_teacher_path', type=str, default="./pre_trained_teacher.pth",
                    help='Path to the state dictionary of the pre-trained teacher. Default: ./pre_trained_teacher.pth')
parser.add_argument('--pre_trained_student_path', type=str, default="./pre_trained_student.pth",
                    help='Path to the state dictionary of the pre-trained student. Default: ./pre_trained_student.pth')
parser.add_argument('--continuous_saving', default=False, action='store_true',
                    help='Save the generator and discriminator model after each epoch? Default: false')
parser.add_argument('--save_state_dir', type=str, default="./",
                    help='Directory where the model states and the semantic information will be saved. Default: ./')
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
    except Exception:
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

        if("LGG" in sample_files):
            LGG_or_HGG_path = os.path.join(sample_path, "LGG")
            LGG = True
        if("HGG" in sample_files):
            LGG_or_HGG_path = os.path.join(sample_path, "HGG")
            HGG = True

        if(LGG and HGG):
            raise Exception("Error: Both LGG and HGG are true.")

        if((not LGG) and (not HGG)):
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
            raise Exception("Error: Could not find ground truth for sample",sample)

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

    # For BraTS, the number of teacher modalities/sequences is 4 and the number of student modalities/sequences is 3.
    num_mods_teacher = 4
    num_mods_student = 3
    
    # Create semantic info.
    semantics = ((set_x, set_y, set_z), num_mods_teacher, num_mods_student)
    
    # Save semantic info.
    np.save(os.path.join(args.save_state_dir,"HAD_Net_semantics.npy"), np.asarray(semantics, dtype=object))
    
    return semantics


def get_dataloaders(list_ids, root_paths,semantics):

    # Create training dataloader.
    data_gen_train = data_generator(list_ids[0], root_paths[0], semantics)
    trainset = torch.utils.data.DataLoader(data_gen_train, batch_size = args.batch_size, shuffle = True, num_workers=args.num_tr_workers)

    print("Created training dataloader.")

    # Create validation dataloader.
    data_gen_val = data_generator(list_ids[1], root_paths[1], semantics)
    valset = torch.utils.data.DataLoader(data_gen_val, batch_size = args.batch_size, shuffle = True, num_workers=args.num_val_workers)

    print("Created validation dataloader.")

    # Create testing dataloader.
    if(args.testing_set):
        data_gen_test = data_generator(list_ids[2], root_paths[2], semantics)
        testset = torch.utils.data.DataLoader(data_gen_test, batch_size = args.batch_size, shuffle = False)
        print("Created testing dataloader.")
        
        return trainset, valset, testset
        
    return trainset, valset, None
    
    
# Create instance of the models and put them on the GPU
def get_models(device):  
    
    # Teacher network. Note: For BraTS, the total number of modalities/sequences is 4. Also, BraTS has 4 segmentation classes.
    teacher_model = Generator(num_mods=4, k=args.init_num_filters, p=args.dropout_prob, num_class=4).to(device)
    print("Created teacher model.")

    # Student network. Note: For BraTS, there is only 1 post-contrast modalities/sequences, so the student recieves 3. 
    # Also, BraTS has 4 segmentation classes.
    generator_model = Generator(num_mods=3, k=args.init_num_filters, p=args.dropout_prob, num_class=4).to(device)
    print("Created generator model (i.e., the student model).")

    # Discriminator network. The discriminator recieves the student input modalities/sequences (i.e., 3 for BraTS) and the 
    # output segmentation map (i.e., 4 classes for BraTS) as input (i.e., a total of 3+4 channels).
    discriminator_model = Discriminator(k_in_init=3+4, k_out_init=args.init_num_filters).to(device)
    print("Created discriminator/critic.")
    
    return teacher_model, generator_model, discriminator_model

# Get losses and softmax:
def get_criteria(device):
    
    # Criterion
    criterion_MSE = torch.nn.MSELoss().to(device)
    weights = torch.tensor([1.0000, 59.4507, 23.3190, 71.8481]) # specifically found for the BraTS 2019 dataset
    criterion_CE = nn.CrossEntropyLoss(weight = weights).to(device)

    # softmax
    n_softmax = nn.Softmax(dim=1).to(device)
    
    return criterion_MSE, criterion_CE, n_softmax

# Initialize the optimizers.
def get_optimizers(generator_model, discriminator_model):

    # Select an optimizer for the generator
    gen_optimizer = torch.optim.Adam(generator_model.parameters(), lr=args.gen_lr, betas=(args.beta_1, args.beta_2))

    # Select an optimizer for the discriminator
    disc_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=args.disc_lr, betas=(args.beta_1, args.beta_2))
    
    return gen_optimizer, disc_optimizer

# Load pre-trained models
def pre_training(teacher_model, generator_model):

    # LOAD PRE-TRAINED TEACHER
    teacher_pretrained_statedict = torch.load(args.pre_trained_teacher_path)
    teacher_model.load_state_dict(teacher_pretrained_statedict)

    # LOAD PRE-TRINAED STUDENT
    student_pretrained_statedict = torch.load(args.pre_trained_student_path)
    generator_model.load_state_dict(student_pretrained_statedict)

    # Freeze teacher weights.
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    return teacher_model, generator_model 
    
# Train for an epoch. 
def training_epoch(models, criteria, optimizers, trainset, device):
    
    # Unpack
    teacher_model, generator_model, discriminator_model = models
    criterion_MSE, criterion_CE  = criteria
    gen_optimizer, disc_optimizer = optimizers
    
    # Counter
    sample_count = 0.0
    
    # Loss tracker
    mean_LS = 0.0
    mean_LHD = 0.0

    # Set to train
    generator_model.train()
    teacher_model.train()
    discriminator_model.train()

    # Go over each batch of the training set
    for data in trainset:

        print("--- SAMPLE",int(sample_count)," ---")

        ###############################################
        ############### GET INPUT DATA ################
        ###############################################        

        # Extract input and target
        x_teacher = data[0].float().to(device)
        x_student = data[1].float().to(device)
        y = data[2].long().to(device)
        sample_name = data[3]

        # Get the teacher output for this sample. This will be the real segmentation map.
        with torch.no_grad():
            real_segmap, real_features = teacher_model(x_teacher)

        ###########################################
        ########### TRAIN THE GENERATOR ###########
        ###########################################

        # zero the generator gradient
        gen_optimizer.zero_grad()

        # Run the input data through the generator
        fake_segmap, fake_features = generator_model(x_student)

        # Feed the disc the "fake" data
        disc_fake_adv = discriminator_model(fake_segmap, x_student, fake_features)

        # Create real and fake labels.
        disc_out_shape = disc_fake_adv.shape
        real = torch.ones(disc_out_shape).to(device)
        fake = torch.zeros(disc_out_shape).to(device)

        # Compute adversarial loss.
        gen_loss_GAN = criterion_MSE(disc_fake_adv, real)

        # Compute voxel-base loss with GROUND TRUTH
        gen_loss_VOX = criterion_CE(fake_segmap, y)

        # Compute TOTAL gen loss, back-propogate, and step generator optimizer forward
        gen_loss = gen_loss_VOX + args.lambda_const*gen_loss_GAN
        gen_loss.backward()
        gen_optimizer.step()

        ###############################################
        ########### TRAIN THE DISCRIMINATOR ###########
        ###############################################

        # zero the discriminator gradient
        disc_optimizer.zero_grad()

        # REMOVE GRADIENT FOR GENERATOR
        fake_segmap = fake_segmap.detach()
        fake_features[0][0] = fake_features[0][0].detach()
        fake_features[0][1] = fake_features[0][1].detach()
        fake_features[1][0] = fake_features[1][0].detach()
        fake_features[1][1] = fake_features[1][1].detach()
        fake_features[2][0] = fake_features[2][0].detach()
        fake_features[2][1] = fake_features[2][1].detach()

        # Feed the disc the "real" data 
        disc_real = discriminator_model(real_segmap, x_student, real_features)
        disc_loss_real = criterion_MSE(disc_real, real)
        print("disc_real =",torch.mean(disc_real).item())

        # Feed the disc the "fake" data 
        disc_fake = discriminator_model(fake_segmap, x_student, fake_features)
        disc_loss_fake = criterion_MSE(disc_fake, fake)
        print("disc_fake =",torch.mean(disc_fake).item())

        # Compute total discriminator loss.
        disc_loss = disc_loss_real + disc_loss_fake

        # Determine if we should update the discriminator for this sample.
        disc_real_mean = torch.mean(torch.ge(disc_real,0.5).float())
        disc_fake_mean = torch.mean(torch.le(disc_fake,0.5).float())
        disc_mean = (disc_real_mean + disc_fake_mean)/2.0

        # Back-propogate the loss and step discriminator optimizer forward, if discriminator performance is
        # under the threshold.
        if(disc_mean <= args.disc_threshold):
            disc_loss.backward()    
            disc_optimizer.step()
            print("Discriminator updated.")

        # Move data from GPU to CPU. This is done in order to prevent a strange CUDA error encountered during training, which 
        # prints the message: "CUDA: an illegal memory access was encountered".
        x_teacher = x_teacher.detach().to('cpu')
        x_student = x_student.detach().to('cpu')
        y = y.detach().to('cpu')
        real_segmap = real_segmap.detach().to('cpu')
        real_features[0][0] = real_features[0][0].detach().to('cpu')
        real_features[0][1] = real_features[0][1].detach().to('cpu')
        real_features[1][0] = real_features[1][0].detach().to('cpu')
        real_features[1][1] = real_features[1][1].detach().to('cpu')
        real_features[2][0] = real_features[2][0].detach().to('cpu')
        real_features[2][1] = real_features[2][1].detach().to('cpu')
        disc_fake_adv = disc_fake_adv.detach().to('cpu')
        real = real.detach().to('cpu')
        fake = fake.detach().to('cpu')
        gen_loss_GAN = gen_loss_GAN.detach().to('cpu')
        gen_loss_VOX = gen_loss_VOX.detach().to('cpu')
        gen_loss = gen_loss.detach().to('cpu')
        fake_segmap = fake_segmap.detach().to('cpu')
        fake_features[0][0] = fake_features[0][0].detach().to('cpu')
        fake_features[0][1] = fake_features[0][1].detach().to('cpu')
        fake_features[1][0] = fake_features[1][0].detach().to('cpu')
        fake_features[1][1] = fake_features[1][1].detach().to('cpu')
        fake_features[2][0] = fake_features[2][0].detach().to('cpu')
        fake_features[2][1] = fake_features[2][1].detach().to('cpu')
        disc_real = disc_real.detach().to('cpu')
        disc_loss_real = disc_loss_real.detach().to('cpu')
        disc_fake = disc_fake.detach().to('cpu')
        disc_loss_fake = disc_loss_fake.detach().to('cpu')
        disc_loss = disc_loss.detach().to('cpu')
        disc_real_mean = disc_real_mean.detach().to('cpu')
        disc_fake_mean = disc_fake_mean.detach().to('cpu')
        disc_mean = disc_mean.detach().to('cpu')
    
        # Update loss trackers.
        mean_LS = mean_LS + gen_loss.item()
        mean_LHD = mean_LHD + disc_loss.item()
        
        # Increment sample counter. 
        sample_count+=1.0
        
    # Find epoch loss.
    mean_LS = mean_LS/sample_count
    mean_LHD = mean_LHD/sample_count

    return (teacher_model, generator_model, discriminator_model), (gen_optimizer, disc_optimizer), (mean_LS, mean_LHD)


# Perform validation for this epoch.
def validation_epoch(generator_model, criterion_CE, valset, n_softmax, running_vm, validate_vm_list, device, epoch):
    
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
    generator_model.eval()
    
    # Go over each batch of the validation set
    for data in valset:
    
        # Extract input and target
        x_teacher = data[0].float().to(device)
        x_student = data[1].float().to(device)
        y = data[2].long().to(device)
        sample_name = data[3]

        # No gradients.
        with torch.no_grad():
            
            # Run the input data through the generator
            fake_segmap, fake_features = generator_model(x_student)
                        
            # Compute loss.
            val_loss = criterion_CE(fake_segmap, y)
            
        # Add loss to sum.
        validation_loss = validation_loss + val_loss.detach().to('cpu').item()
                            
        # Get the output image.
        output_image = n_softmax(fake_segmap)
        output_image = torch.argmax(output_image, dim=1)
        
        # Compute Dice score
        dice_NET, dice_ED, dice_ET, dice_WT, dice_TC = multi_class_dice(output_image, y)
        dice_scores_NET = dice_scores_NET + dice_NET.item()
        dice_scores_ED = dice_scores_ED + dice_ED.item()
        dice_scores_ET = dice_scores_ET + dice_ET.item() 
        dice_scores_WT = dice_scores_WT + dice_WT.item()
        dice_scores_TC = dice_scores_TC + dice_TC.item()
        
        # Increment counter
        sample_count = sample_count + 1.0
        
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
    
    # Get dataloaders for the training, validation, and testing sets.
    trainset, valset, testset = get_dataloaders((training_ids, validation_ids, testing_ids),(args.training_root_path, args.validation_root_path, args.testing_root_path),semantics)

    # Create instance of the models and put them on the GPU
    teacher_model, generator_model, discriminator_model = get_models(device)
              
    # Get losses and softmax:
    criterion_MSE, criterion_CE, n_softmax = get_criteria(device)

    # Initialize the optimizers.
    gen_optimizer, disc_optimizer = get_optimizers(generator_model, discriminator_model)
              
    # Load pre-trained models
    teacher_model, generator_model = pre_training(teacher_model, generator_model)
              
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
        models, optimizers, training_losses = training_epoch((teacher_model, generator_model, discriminator_model), (criterion_MSE, criterion_CE), (gen_optimizer, disc_optimizer), trainset, device)
              
        teacher_model, generator_model, discriminator_model = models
        gen_optimizer, disc_optimizer = optimizers
        mean_LS, mean_LHD = training_losses      

        # VALIDATION
        print(" ")
        print("Beginning validation for epoch",epoch,".")
        print(" ")
              
        # Validation for this epoch.
        mean_val_loss, mean_dice_scores, vm_info = validation_epoch(generator_model, criterion_CE, valset, n_softmax, running_vm, validate_vm_list, device, epoch)
              
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
            torch.save(generator_model.state_dict(), "HAD_Net_BEST_VAL_VM.pth")
            print("Model parameters saved: best VALIDATE_VM.")

        # Save generator and discriminator model and optimizer.
        if(args.continuous_saving):
            torch.save(generator_model.state_dict(),"gen_model.pth")
            torch.save(gen_optimizer.state_dict(), "gen_optim.pth") 
            torch.save(discriminator_model.state_dict(), "disc_model.pth")
            torch.save(disc_optimizer.state_dict(), "disc_optim.pth") 
            print("State dictionaries saved for this epoch.")

        # Go back to original dir.
        os.chdir(original_dir)

        # Epoch complete. 
        print("Epoch", epoch, "complete.")
        print("------------------------------------------------------------------------")
              
    ###########################################
    ################# TESTING #################
    ###########################################
    
    # If there is a local testing set, run the model on it.
    if(args.testing_set):
       
        # Verbose
        print("Running best model on the local testing set.")

        # Load best model
        original_dir = os.getcwd()
        os.chdir(args.save_state_dir)
        generator_model.load_state_dict(torch.load("HAD_Net_BEST_VAL_VM.pth"))
        os.chdir(original_dir)

        # Run model on testing set.
        mean_test_loss, mean_dice_scores, vm_info = validation_epoch(generator_model, criterion_CE, testset, n_softmax, 0.0, [], device, epoch)

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
