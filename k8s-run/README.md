RuntimeError: can not read labels file "/rsrch7/wulab/Mori/imageNet/ILSVRC/Data/CLS-LOC/labels.txt"
only with old image and cuda 12, all packages can be installed.

Error:
RuntimeError: can not read labels file "/rsrch7/wulab/Mori/imageNet/ILSVRC/Data/CLS-LOC/labels.txt"
RuntimeError: can not read labels file "/rsrch7/wulab/Mori/imageNet/ILSVRC/Data/CLS-LOC/labels.txt"
Finally 0 is done, I had to make artificial val folder based on train folder as Dino expected. (The original val did not have folders just the image.)



# Setting up image:
1. having images 
2. test installing mmcv on my .py code

with 11 does not wok but mmcv could be intalled!
      nodeSelector:
        nvidia.com/cuda.runtime.major: "12"
        "nvidia.com/gpu.present": "true"
        "nvidia.com/gpu.machine": "DGXA100-920-23687-2530-000"




Let's build my dockerfile first based on dino v2 requrements and see if everything works.
I probably need to install these on my .py file to make sure they are compatible with cuda and more.

Can we define with cuda to run in jupyterhub?

docker tag SOURCE_IMAGE[:TAG] hpcharbor.mdanderson.edu/foundino/REPOSITORY[:TAG]

docker push hpcharbor.mdanderson.edu/foundino/REPOSITORY[:TAG]

docker build -t foundino:msalehjahromi .
docker build --platform linux/x86_64 -t foundino:msalehjahromi .

docker login hpcharbor.mdanderson.edu
docker tag foundino:msalehjahromi hpcharbor.mdanderson.edu/foundino/foundino:msalehjahromi
docker push hpcharbor.mdanderson.edu/foundino/foundino:msalehjahromi

docker pull hpcharbor.mdanderson.edu/foundino/foundino@sha256:53e887c22ff97807970029413d77db1d0f3c0ac959227e59250d93e6138f038d

hpcharbor.mdanderson.edu/foundino/foundino@sha256:53e887c22ff97807970029413d77db1d0f3c0ac959227e59250d93e6138f038d

# #########################################################
job-runner.sh xxx.yaml
kubectl delete job -n yn-gpu-workload msalehjahromi-gpu-xxx
kubectl apply -f x.yaml




List and describe the PVCs with the following commands:
```sh
kubectl get pvc -n yn-gpu-workload -l k8s-user=msalehjahromi
kubectl describe pvc msalehjahromi-gpu-rsrch7-home-ip-rsrch -n yn-gpu-workload
```

## Environment Configuration
```sh
export PATH="/rsrch1/ip/msalehjahromi/.kube:$PATH"
source ~/.bashrc
```

## Useful Kubernetes (K8s) Commands
- View jobs:
```sh
kubectl get jobs
```
- Switch context:
```sh
kubectl config use-context msalehjahromi_yn-gpu-workload@research-prd
```
- Delete a job:
```sh
kubectl delete job -n yn-gpu-workload msalehjahromi-gpu-xxx
```

## YAML Configurations for Different GPU Types
```sh
        "nvidia.com/gpu.machine": "DGXH100"
        "nvidia.com/gpu.machine": "DGXA100-920-23687-2530-000"
```

## nnUNetor 2000 epochs!
```sh
nnUNetv2_train dataset_id 3d_fullres num_fold -tr nnUNetTrainer_2000epochs
```


###########
batch 4
vtorch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 580.00 MiB (GPU 0; 39.38 GiB total capacity; 6.18 GiB already allocated; 191.12 MiB free; 6.30 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

batch 32
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB (GPU 0; 39.38 GiB total capacity; 7.88 GiB already allocated; 44.69 MiB free; 7.96 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB (GPU 0; 39.38 GiB total capacity; 14.38 GiB already allocated; 44.69 MiB free; 14.45 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB (GPU 0; 39.38 GiB total capacity; 1.12 GiB already allocated; 8.44 MiB free; 1.12 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF




# first error:
  File "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader/dinov2/data/loaders.py", line 52, in _parse_dataset_str
    assert key in ("root", "extra", "split")
AssertionError

# second error
  File "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader/dinov2/data/collate.py", line 13, in collate_data_and_cast
    n_global_crops = len(samples_list[0][0]["global_crops"])
TypeError: 'Image' object is not subscriptable


changes
## 1_train.py
## CT_net.py
## extended.py   __getitem__

## loaders.py
    1. from .datasets import CTDataset
    2. _parse_dataset_str

## __init__.py
from .ct_net import CTDataset

## dinov2/configs/train/vitl16_short.yaml
  dataset_path: ImageNet:split=TRAIN      dataset_path: CTDataset:h5_file_path=/rsrch7/home/ip/msalehjahromi/datasets/ct_scans.h5:patch_size=512,512,3

## dinov2/configs/ssl_default_config.yaml
dataset_path: ImageNet:split=TRAIN
dataset_path: CTDataset:h5_file_path=/rsrch7/home/ip/msalehjahromi/datasets/ct_scans.h5:patch_size=512,512,3

## 





In the get_image_data method, ensure that the dimensions of the CT data are sufficient for the desired patch_size.



# Loss goes down quickly
1. Limited Variability in CT Scans
    Spatial Augmentations: Apply random rotations, flips, elastic deformations, scaling, and cropping.
    Intensity Transformations: Add Gaussian noise, random brightness/contrast adjustments, or histogram equalization.
    Domain-Specific Augmentations: Simulate variations in Hounsfield Units (e.g., altering windowing for "soft tissue" or "bone").
    CutMix or MixUp: Blend images together or use partial image mixing to force the model to generalize.

2. Lack of Augmentation Diversity
    Ensure you are using strong augmentations like those in the original DINO paper:
    Multi-crop strategy (e.g., 2 large crops + 8 smaller crops).
    Color jittering, Gaussian blur, and solarization (if these make sense for CT images).
    Adapt these to 3D if your data is volumetric:
    Use random 3D rotations, flipping along axes, and cropping slices from different orientations.

5. Learning Rate and Optimization
    If the learning rate is too high, the model may quickly converge to a suboptimal solution, causing the loss to plateau.
    Solution:
    Reduce the learning rate and use a warm-up schedule.
    Regularize training with weight decay or gradient clipping.
    
6. Feature Collapse
    In self-supervised learning, there’s a risk of the model collapsing to trivial solutions where all features are identical.
    Solution:
    Ensure the implementation of DINO avoids collapse. For instance, the use of a momentum teacher and centering/sharpening mechanisms should be in place.
    Increase batch size or number of crops to stabilize training.