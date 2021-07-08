# HAD-Net: A Hierarchical Adversarial Knowledge Distillation Network for Improved Enhanced Tumour Segmentation Without Post-Contrast Images

## Paper
![Image Not Found](./Qualitative_Results.png)


[HAD-Net: A Hierarchical Adversarial Knowledge Distillation Network for Improved Enhanced Tumour Segmentation Without Post-Contrast Images](https://arxiv.org/abs/2103.16617)  
Medical Imaging with Deep Learning (MIDL), 2021

If you use any resources from this repository or find it useful for your research, please cite our paper:

Saverio Vadacchino, Raghav Mehta, Nazanin Mohammadi Sepahvand, Brennan Nichyporuk, James J. Clark, Tal Arbel. "HAD-Net: A Hierarchical Adversarial Knowledge Distillation Network for Improved Enhanced Tumour Segmentation Without Post-Contrast Images". In MIDL 2021. https://arxiv.org/abs/2103.16617

```
@article{vadacchino2021had,
  title={HAD-Net: A Hierarchical Adversarial Knowledge Distillation Network for Improved Enhanced Tumour Segmentation Without Post-Contrast Images},
  author={Vadacchino, Saverio and Mehta, Raghav and Sepahvand, Nazanin Mohammadi and Nichyporuk, Brennan and Clark, James J and Arbel, Tal},
  journal={arXiv preprint arXiv:2103.16617},
  year={2021}
}
```

## Abstract

Segmentation of enhancing tumours or lesions from MRI is important for detecting new disease activity in many clinical contexts. However, accurate segmentation requires the inclusion of medical images (e.g., T1 post-contrast MRI) acquired after injecting patients with a contrast agent (e.g., Gadolinium), a process no longer thought to be safe. Although a number of modality-agnostic segmentation networks have been developed over the past few years, they have been met with limited success in the context of enhancing pathology segmentation. In this work, we present HAD-Net, a novel offline adversarial knowledge distillation (KD) technique, whereby a pre-trained teacher segmentation network, with access to all MRI sequences, teaches a student network, via hierarchical adversarial training, to better overcome the large domain shift presented when crucial images are absent during inference. In particular, we apply HAD-Net to the challenging task of enhancing tumour segmentation when access to post-contrast imaging is not available. The proposed network is trained and tested on the BraTS 2019 brain tumour segmentation challenge dataset, where it achieves performance improvements in the ranges of 16% - 26% over (a) recent modality-agnostic segmentation methods (U-HeMIS, U-HVED), (b) KD-Net adapted to this problem, (c) the pre-trained student network and (d) a non-hierarchical version of the network (AD-Net), in terms of Dice scores for enhancing tumour (ET). The network also shows improvements in tumour core (TC) Dice scores. Finally, the network outperforms both the baseline student network and AD-Net in terms of uncertainty quantification for enhancing tumour segmentation based on the BraTS 2019 uncertainty challenge metrics.


## Overview
This repository contains the source code for brain tumour segmentation when post-contrast images/sequences are unavailable during inference using the method detailed in [HAD-Net](https://arxiv.org/abs/2103.16617). It also contains pre-trained models which were trained on the BraTS 2019 dataset. Note that this was implemented in PyTorch.

## Requirements
* CUDA compatible GPU. For training, the GPU needs to have at least 18GB of memory. However, for inference, the GPU needs only 8GB. 
* Python 3.6.9
* PyTorch 1.8.0, which can be installed with the following pip command:
```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* The libraries listed in the requirements.txt file, which can be downloaded with the following command:
```bash
pip install -r requirements.txt
```
* BraTS dataset

## Usage

### Data-Preprocessing (Training)
1) Download the BraTS dataset (both Training and Validation). For example, the BraTS 2019 dataset can be found [here](https://www.med.upenn.edu/cbica/brats2019/registration.html).
2) Pre-process the training data by running the script ``data_preprocessor.py`` found in the utils folder.  This script crops the input images, flips certain axes, and creates a brain mask. This script can be run with the following command: 
```bash
python data_preprocessor.py --root-dir ./dataset/original_data --proc-dir ./dataset/processed_data --num-workers 4
```
Note: the ``root-dir`` argument defines the directory where the data to be processed is stored, the ``proc-dir`` argument defines the directory in which the processed data will be stored, and the ``num-workers`` argument defines the number of workers you wish to use to perform the pre-processing. 

Furthermore, note that this script should only be used to pre-process the training data. The script used to pre-process the inference data (i.e., the validation data) is detailed in the Inference section of this README.

Finally, note that this script was written specifically for pre-processing BraTS (2019) data.

### Pre-Training
Prior to training HAD-Net, the student network and teacher network need to be pre-trained independently. In order to perform pre-training, one needs:
* The pre-processed data which results from running ``data_preprocessor.py``.
* A list containing the names (i.e., ids) of the samples used in the training set, saved as a ``.npy`` file. 
* A list containing the names (i.e., ids) of the samples used in the (local) validation set, saved as a ``.npy`` file.
* A list containing the names (i.e., ids) of the samples used in the (local "held-out") testing set, saved as a ``.npy`` file, if one is used.

Note that there mustn't be any overlap in the names contained within each of the id files, as the training scripts will terminate if they detect a data leak. Note that the example of a sample name/id used in the BraTS 2019 dataset is "BraTS19_TCIA02_290_1".

As pre-training progresses (for both the student and teacher networks), the state dictionaries of the best models will be saved. The "best" pre-trained teacher model will be saved as ``Pre_Trained_Teacher_BEST_VAL_VM.pth`` and the "best" pre-trained student model will be saved as ``Pre_Trained_Student_BEST_VAL_VM.pth``. These will be required to train HAD-Net. Note that the "best" model is that which achieves the best performance (i.e., the best segmentations) on the validation set. Both ``Pre_Trained_Teacher_BEST_VAL_VM.pth`` and ``Pre_Trained_Teacher_BEST_VAL_VM.pth`` will be saved to the directory specified by the user, via the ``save_state_dir`` argument. 

#### Teacher Network
Pre-train the teacher network by running the script ``Teacher_Pretraining.py``. This script has several arguments which can be used to alter certain hyper-parameters; however, it is important to note that the default values of these arguments are those used in our paper. While the majority of these default values do not need to be altered, some do (like those that define file paths). Those that will likely need to be altered are:
* ``GPU_index``, which defines the GPU to use during training.
* ``training_root_path``, which defines the path to pre-processed training data.
* ``validation_root_path``, which defines the path to pre-processed (local) validation data. Note that this is likely the same as the training data path.
* ``testing_root_path``, which defines the path to pre-processed (local) testing data. Note that this is likely also the same as the training data path.
* ``training_ids_path``, which defines the path to the file which lists the name of the data samples used in the training set.
* ``validation_ids_path``, which defines the path to the file which lists the name of the data samples used in the (local) validation set.
* ``testing_ids_path``, which defines the path to the file which lists the name of the data samples used in the (local) testing set.
* ``save_state_dir``, which defines the path to the directory to which the model state dictionaries will be saved during training. 

Also note that the arguments include two flags:
* ``testing_set``, which indicates the use of a local testing set, on which the model is evaluated at the very end of training. Not including this in the arguments indicates the absence of a local testing set, while the inclusion of the argument "--testing_set" indicates the presence of one. 
* ``continous_saving``, which indicates the desire to save a checkpoint after every training epoch. Note including this in the arguments indicates that one does not want this saving to take place, while the inclusion of the argument activates this saving mechanic. 

Therefore, one can pre-train the teacher network (with a testing set and continuous saving active) with a variation of the following command:
```bash
python Teacher_Pretraining.py --GPU_index 0 --training_root_path ./dataset/processed_data/Training/BRATS --validation_root_path ./dataset/processed_data/Training/BRATS --testing_set --testing_root_path ./dataset/processed_data/Training/BRATS --training_ids_path ./dataset/ids/training_ids.npy --validation_ids_path ./dataset/ids/validation_ids.npy --testing_ids_path ./dataset/ids/testing_ids.npy --continuous_saving --save_state_dir ./save_states
```

#### Student Network
Pre-train the student network by running the script ``Student_Pretraining.py``.

This script uses the same arguments as ``Teacher_Pretraining.py``. Therefore, one can pre-train the student network (with a testing set and continuous saving active) with a variation of the following command:

```bash
python Student_Pretraining.py --GPU_index 0 --training_root_path ./dataset/processed_data/Training/BRATS --validation_root_path ./dataset/processed_data/Training/BRATS --testing_set --testing_root_path ./dataset/processed_data/Training/BRATS --training_ids_path ./dataset/ids/training_ids.npy --validation_ids_path ./dataset/ids/validation_ids.npy --testing_ids_path ./dataset/ids/testing_ids.npy --continuous_saving --save_state_dir ./save_states
```

### Training

Once pre-training is complete, one can start training HAD-Net, using ``HAD_Net.py``. In addition to what was required to do pre-training (i.e., the pre-processed data and the list of samples ids), to perform training with HAD-Net, one needs:
* The state dictionary of the pre-trained teacher network.
* The state dictionary of the pre-trained student network.

Naturally, HAD-Net should be trained (and validated) on the same set of examples used to train the student and teacher networks during pre-training.

The script used for training HAD-Net uses many of the same arguments used in the script used for pre-training. It also has a number of additional arguments which can be used to alter certain hyper-parameters; however, it is important to note that the default values of these arguments are those used in our paper. While the majority of these default values do not need to be altered, some do. In addition to the arguments listed in the pre-training section of this README (i.e., ``GPU_index``, ``training_root_path``, etc.), those that will likely need to be altered are:
* ``pre_trained_teacher_path``, which defines the path to the state dictionary of the pre-trained teacher network.
* ``pre_trained_student_path``, which defines the path to the state dictionary of the pre-trained student network.

Like the pre-training scripts, ``HAD-Net.py`` has the two flag arguments ``testing_set`` and ``continuous_saving`` while indicates the presence of a local "held-out" testing set and the desire to continuously save checkpoints, respectively.

Therefore, one can train HAD-Net (with a testing set and continuous saving active) with a variation of the following command:

```bash
python HAD_Net.py --GPU_index 0 --training_root_path ./dataset/processed_data/Training/BRATS --validation_root_path ./dataset/processed_data/Training/BRATS --testing_set --testing_root_path ./dataset/processed_data/Training/BRATS --training_ids_path ./dataset/ids/training_ids.npy --validation_ids_path ./dataset/ids/validation_ids.npy --testing_ids_path ./dataset/ids/testing_ids.npy --pre_trained_teacher_path ./save_states/Pre_Trained_Teacher_BEST_VAL_VM.pth --pre_trained_student_path ./save_states/Pre_Trained_Student_BEST_VAL_VM.pth --continuous_saving --save_state_dir ./save_states
```

At the start of training, this script will save a file named ``HAD_Net_Semantics.npy``, which will be required to perform inference. Effectively, the inference script uses the information stored withing ``HAD_Net_Semantics.npy`` to ensure that the inference samples are made to be the same size as those used while trianing HAD-Net. Furthermore, as training progresses, the state dictionary of the best HAD-Net model will be saved ``HAD_Net_BEST_VAL_VM.pth``. Note that the "best" model is that which achieves the best performance (i.e., the best segmentations) on the validation set. This model can then be used to perform inference without post-contrast images. Both ``HAD_Net_Semantics.npy`` and ``HAD_Net_BEST_VAL_VM.pth`` will be saved in the directory specified by the ``save_state_dir`` argument.

### Inference

Once training is complete, one can use the "best" HAD-Net model produced during training to perform inference on new samples, without the need for post-contrast images. This can be done by running the ``inference.py`` script. 

Prior to running the script, one needs to pre-process the inference samples with the data pre-processing script ``data_preprocessor_inference.py``, which can be found in the utils folder. Note that this script is slightly different from the script used to pre-process the training data (i.e., those with ground truth segmentations). However, similar to the ``data_preprocessor.py`` script, the ``data_preprocessor_inference.py`` script crops the input images, flips certain axes, and creates a brain mask. Ultimately, one can run the inference data pre-processor with the following command: 
```bash
python data_preprocessor_inference.py --root-dir ./dataset/original_inference_data --proc-dir ./dataset/processed_inference_data --num-workers 4
```
Note: the ``root-dir`` argument defines the directory where the inference data to be processed is stored, the ``proc-dir`` argument defines the directory in which the processed inference data will be stored, and the ``num-workers`` argument defines the number of workers you wish to use to perform the pre-processing.

Furthermore, note that this script was written specifically for pre-processing BraTS (2019) data.

In all, to perform inference one needs:
* The pre-processed inference data which results from running ``data_preprocessor_inference.py``.
* The semantic information saved by ``HAD-Net.py`` at the start of training.
* A trained HAD-Net model.

The ``inference.py`` script uses the following arguments:
* ``GPU_index``, which defines the GPU to use during inference.
* ``type``, which defines the type of model being used for inference (i.e., a ``student`` model which receives only pre-contrast sequences, or a ``teacher`` model which receives all sequences).
* ``inference_proc_root_path``, which defines the path to pre-processed inference data.
* ``inference_original_root_path``, which defines the path to original (unprocessed) inference data.
* ``inference_ids_path``, which defines the path to the file which lists the name of the inference samples.
* ``batch_size``, which defines the batch size to use when performing inference.
* ``init_num_filters``, which defines the initial number of output channels (i.e., ``k``) used in the model. Note that if this is changed from the default value, it must match the value of ``k`` used during both the training of HAD-Net and the pre-training of the student and teacher models.
* ``model_path``, which defines the path to the trained HAD-Net model (i.e., the model trained via ``HAD_Net.py``).
* ``semantics_path``, which defines the path to the semantic information saved by ``HAD_Net.py`` at the start of training.
* ``output_dir``, which defines the path to the directory to which the output segmentation maps will be saved. 

Therefore, one can perform inference with a variation of the following command:
```bash
python HAD_Net.py --GPU_index 0 --inference_proc_root_path ./dataset/processed_inference_data --inference_original_root_path ./dataset/original_inference_data --inference_ids_path ./dataset/ids/inference_ids.npy --model_path ./save_states/HAD_Net_BEST_VAL_VM.pth --semantics_path ./save_states/HAD_Net_Semantics.npy --output_dir ./output_inference
```

The script will save the output segmentations to the directory specified by the ``output_dir`` arguments. These segmentations will be saved as ``.nii.gz`` files. Note that the ``inference.py`` script performs post-processing on the segmentations prior to saving, and ensures that they are same size as, and are aligned with, the original input images (prior to pre-processing).

Note that the ``save_segmenation`` section of the script was written specifically for post-processing the BraTS (2019) dataset.

### Trained Models

This repository also contains trained models (trained on the BraTS 2019 dataset) for HAD-Net, the pre-trained Student network, and the pre-trained Teacher network. These can be found in the ``trained_examples`` folder, along with the semantic information file used during training. Note that these are the models used to obtain the results presented in our paper [HAD-Net](https://arxiv.org/abs/2103.16617).