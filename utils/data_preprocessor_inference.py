import os
import time
from threading import Timer
import multiprocessing
from multiprocessing import Process, JoinableQueue

import nibabel as nib
from os import listdir, makedirs
from os.path import join, exists, realpath

from scipy import ndimage
import numpy as np
import traceback
import argparse
import sys
import re

path = realpath(__file__)
sys.path.append(path[:path.find('/image')])

import constants

# Argument Parser
parser = argparse.ArgumentParser()

# Required Arguments
parser.add_argument('--root-dir', type=str, default="./BraTS19/Validation_Original")
parser.add_argument('--proc-dir', type=str, default="./BraTS19/Validation_Processed")
parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count())

args = parser.parse_args()


def build_leaf_index(abs_path, suffix_dict):
    index = {}
    filenames = list(filter(lambda x: x.endswith(constants.EXTENSIONS), listdir(abs_path)))
    for key in suffix_dict:
        try:
            flnm = next(filter(lambda fn: fn.endswith(suffix_dict[key]), filenames))
            index[key] = join(abs_path, flnm)
        except BaseException:
            pass
    return index


def load(path):
    if path.endswith(constants.EXTENSIONS_NIB):
        data = nib.load(path).get_fdata()
    else:
        raise NotImplementedError

    if data.dtype != np.float64:
        data = data.astype(np.float64)
    return data


# get tuple to slice into brain region
def get_slice_tpl(mask_data):
    nonzero = np.nonzero(mask_data)
    amin = np.amin(nonzero, axis=1)
    amax = np.amax(nonzero, axis=1) + 1
    slice_tpl = tuple(np.s_[l:u] for l, u in zip(amin, amax))
    return slice_tpl


# uint16: 0 to 65535
# prevent loss of precision when converting to uint16
def rescale(data, is_brain_data):
    amin = np.amin(data[is_brain_data])
    np.subtract(data, amin, out=data, where=is_brain_data)
    amax = np.amax(data[is_brain_data])
    if amax > 0:
        scale = 65535 / amax
        np.multiply(data, scale, out=data, where=is_brain_data)
        np.around(data, out=data)


def proc_helper(index, trial, subj, tmpt):
    # brats data size: (155, 240, 240)
    mask_data = np.ones(((155, 240, 240)))
    data_list = []
    for img_type in index:
        data = load(index[img_type])

        data = np.transpose(data)
        data = data[:,::-1,:]

        if (img_type in constants.BRATS_SEQUENCES) or (img_type in constants.SEQUENCES):
            # create adhoc brain mask
            mask_data[data==0] = 0
        elif img_type in constants.BRATS_LABELS:
            # BRATS Labels: 0, 1, 2, 4; Replace with: 0, 1, 2, 3
            data[data==4] = 3
        else:
            raise Exception
        data_list.append((img_type, data))

    # only save mask/data containing brain data (via slice_tpl)
    slice_tpl = get_slice_tpl(mask_data)
    mask_data = mask_data[slice_tpl]
    for img_type, data in data_list:
        data = data[slice_tpl]
        # brain mask already applied -- so no need to reapply it here (brain mask should be applied otherwise)
        if img_type in constants.BRATS_LABELS:
            data = data.astype(np.uint8)
        elif (img_type in constants.BRATS_SEQUENCES) or (img_type in constants.SEQUENCES):
            
            # Check for negative values in the image, as these would take on values of 2^16 -1 after conversion.
            # If there are negative values, give user a warning and apply an absolute value operation to the image.
            if((data < 0).any()):
                print("Warning: image",img_type, "for subject", subj, "has negative values. It is recommended that the user look into this. Note that an absolute value operation will subsequently be applied to the image to prevent overflow.")
                data = np.absolute(data)
            
            # BRATS data already in uint16 form -- no need to rescale (otherwise, consider rescaling -- see function above)
            data = data.astype(np.uint16)
        else:
            raise NotImplementedError

        # write data to disk
        makedirs(join(args.proc_dir, trial, subj, tmpt), exist_ok=True)
        np.save(join(args.proc_dir, trial, subj, tmpt, img_type), data)

    # save brain mask to disk
    mask_data = mask_data.astype(np.uint8)
    makedirs(join(args.proc_dir, trial, subj, tmpt), exist_ok=True)
    np.save(join(args.proc_dir, trial, subj, tmpt, constants.MASK), mask_data)


def proc(sequence_dir, label_dir, mask_dir, sequence_suffix, label_suffix, mask_suffix, trial, subj, tmpt):
    
    # check if it is a directory
    if(not os.path.isdir(sequence_dir)):
        return
    
    # build sequence index
    sequence_suffix_dict = {neurorx_seqn: brats_seqn + sequence_suffix for neurorx_seqn, brats_seqn in zip(constants.SEQUENCES, constants.BRATS_SEQUENCES)}
    sequence_index = build_leaf_index(sequence_dir, sequence_suffix_dict)

    # build label index
    label_suffix_dict = {labl: labl + label_suffix for labl in constants.BRATS_LABELS}
    label_index = build_leaf_index(label_dir, label_suffix_dict)

    # if sequence/label/newt2 pre-processing is not complete --> redo
    index = {**sequence_index, **label_index}
    if not all(exists(join(args.proc_dir, trial, subj, tmpt, img_type+constants.EXTENSION_NPY)) for img_type in index):
        # load, process, and write sequence/label data
        proc_helper(index, trial, subj, tmpt)


q = JoinableQueue()

def do_work(item):
    tmpt_dir, subj, tmpt = item
    proc(tmpt_dir, tmpt_dir, tmpt_dir, constants.SUFFIX_BRATS,
         constants.SUFFIX_BRATS, constants.SUFFIX_BRATS,
         constants.BRATS, subj, tmpt)

for subj in listdir(args.root_dir):
    subj_dir = join(args.root_dir, subj)
    item = (subj_dir, subj, "LGG_or_HGG")
    q.put(item)


def worker():
    while True:
        item = q.get()
        if item is None:
            break
        try:
            do_work(item)
        except:
            traceback.print_exc()
        q.task_done()


processes = []
for i in range(args.num_workers):
    p = Process(target=worker)
    p.start()
    processes.append(p)


queue_init_size = q.qsize()
s_time = time.time()
def progress_meter():
    queue_size = q.qsize()
    print('Time Elapsed: {}, Progress: {}%'.format(
        round(time.time() - s_time),
        round((1 - queue_size / queue_init_size) * 100, 2)))
    global tmr
    tmr = Timer(10., progress_meter)
    tmr.start()


progress_meter()
q.join()
tmr.cancel()


for i in range(args.num_workers):
    q.put(None)
for p in processes:
    p.join()
