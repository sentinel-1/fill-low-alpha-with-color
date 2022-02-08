#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime,timedelta
nb_st = datetime.utcnow()
print(f"\nNotebook START time: {nb_st} UTC\n")


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


# In[3]:


IMG_ROOT = Path('data/sample-images')
assert IMG_ROOT.is_dir()
print("Available samples:")
print(sorted([sample_dir.stem for sample_dir in IMG_ROOT.glob('sample-[0-9]*')]))


# In[4]:


for sample_dir in sorted(IMG_ROOT.glob('sample-[0-9]*')):
    for img_name in ['color.jpg','mask.png','original.jpg']:
        if not (sample_dir/img_name).exists():
            raise Exception(f"MISSING: {(sample_dir/img_name).relative_to(IMG_ROOT)}")
else:
    print("All images are in place.")


# In[5]:


def visualize_result(image_arr, result_arr, mask_arr, image_name):
    fontdict = {'color':'white', 'fontsize':14, 'fontweight':'bold'}
    fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(16,16))
    fig.set_facecolor("black")
    for row in ax:
        for ax_i in row:
            ax_i.set_axis_off()
    ax[0,0].imshow(np.transpose(image_arr.cpu().numpy(), [1, 2, 0]))
    ax[0,0].set_title(f"Original ({image_name}):",fontdict=fontdict)
    ax[0,1].imshow(np.transpose(np.concatenate([image_arr.cpu().numpy(),mask_arr.unsqueeze(0).cpu().numpy()]), [1, 2, 0]))
    ax[0,1].set_title("Original with alpha mask:",fontdict=fontdict)
    ax[1,0].imshow(np.transpose(result_arr.cpu().numpy(), [1, 2, 0]))
    ax[1,0].set_title(f"Color Corrected ({image_name}):",fontdict=fontdict)
    ax[1,1].imshow(np.transpose(np.concatenate([result_arr.cpu().numpy(),mask_arr.unsqueeze(0).cpu().numpy()]), [1, 2, 0]))
    ax[1,1].set_title("Color Corrected with alpha mask:",fontdict=fontdict)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# In[6]:


from lib.boundary_pixel_extend import FillLowAlphaWithColor


# In[7]:


import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# In[8]:


for USE_GPU in (False, True,):
    print(f"\n\nINFO: **Testing USE_GPU={USE_GPU} case**\n\n", flush=True)
    if USE_GPU and not torch.cuda.is_available():
        print(f"WARNING: USE_GPU={USE_GPU}, but cuda cappable device is not available on your machine. Skipping GPU part...", file=sys.stderr, flush=True)
        continue
        
    transform = FillLowAlphaWithColor(gpu=USE_GPU)
    # print("INFO: Using image transformer: ", transform, flush=True)


    n_img_processed = 0
    processing_time_total = timedelta(seconds=0)
    min_td = timedelta(seconds=100000)
    max_td = timedelta(seconds=0)
    stats = {}

    for sample_dir in sorted(IMG_ROOT.glob('sample-[0-9]*')):
        with rasterio.open(sample_dir/'original.jpg') as img:
            image_arr = img.read()
        with rasterio.open(sample_dir/'mask.png') as img:
            mask_arr = img.read()
        image_arr = torch.as_tensor(image_arr)
        mask_arr = torch.as_tensor(mask_arr.squeeze())
        orig_image_arr = image_arr.clone().detach()

        if image_arr.shape[-2:] != mask_arr.shape:
            print(f"ERROR: {sample_dir}:"
                  f" Image size {image_arr.shape[-2]}x{image_arr.shape[-1]} "
                  f"and mask size {mask_arr.shape[0]}x{mask_arr.shape[1]} do not match! Skipping...",
                  file=sys.stderr, flush=True)
            continue
        else:
            print(f"INFO: {sample_dir}: Image size {tuple(image_arr.shape[-2:])} and mask size {tuple(mask_arr.shape)} match. Processing...", flush=True)
        processing_start_time = datetime.now()

        result_arr, mask_arr = transform(image_arr, mask_arr)

        processing_end_time = datetime.now()
        n_img_processed += 1
        processing_td = processing_end_time - processing_start_time
        processing_time_total += processing_td
        stats[sample_dir.stem] = processing_td
        if min_td > processing_td:
            min_td = processing_td
        if max_td < processing_td:
            max_td = processing_td
        print(f"INFO: {sample_dir}: Done. Image processing time on {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}: {processing_td}", flush=True)

        visualize_result(orig_image_arr, result_arr, mask_arr, image_name=sample_dir.stem)

    print(f"INFO: Processed {n_img_processed} images", flush=True)
    print(f"INFO: Processing times per sample:", flush=True)
    for k in stats:
        print(f"       - [{'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}] {k}: {stats[k]}", flush=True)
    print(f"INFO: Summary of processing times on {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}:", flush=True)
    print(f"       - {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'} minimum: {min_td}", flush=True)
    print(f"       - {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'} maximum: {max_td}", flush=True)
    print(f"       - {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'} avgerage: {processing_time_total/n_img_processed}", flush=True)
    


# In[9]:


print(f"\n ** Total Elapsed time: {datetime.utcnow() - nb_st} ** \n")
print(f"Notebook END time: {datetime.utcnow()} UTC\n")


# In[ ]:




