
# 1. bashrc
vim ~/.bashrc
export PATH="/rsrch1/ip/msalehjahromi/.kube:$PATH"
source ~/.bashrc

# 2.
export KUBECONFIG=/rsrch1/ip/msalehjahromi/.kube/config

# 3.
env | grep -i proxy

export NO_PROXY=10.113.120.23,localhost,127.0.0.1
export no_proxy=10.113.120.23,localhost,127.0.0.1

# 4. running
kubectl apply -f multiple_GPUs.yaml --v=9


# Run & Delete
kubectl apply -f multiple_GPUs.yaml
kubectl delete job -n yn-gpu-workload msalehjahromi-torchrun-dl5-b8
kubectl delete job -n yn-gpu-workload msalehjahromi-torchrun-dl4-b16


 

# Seq:

  File "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader3/dinov2/data/datasets/extended.py", line 56, in __getitem__
    image, target = self.transforms(image, target)
  File "/rsrch1/ip/msalehjahromi/.local/lib/python3.10/site-packages/torchvision/datasets/vision.py", line 94, in __call__
    input = self.transform(input)
  File "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader3/dinov2/data/augmentations.py", line 53, in __call__
    im1 = self.random_resized_crop(
  File "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader3/dinov2/data/augmentations.py", line 32, in random_resized_crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(
NameError: name 'transforms' is not defined


# 224

1. dinov2/models/__init__.py
def build_model(args, only_teacher=False, img_size=224):

2. dinov2/layers/patch_embed.py
  def __init__(
      self,
      img_size: Union[int, Tuple[int, int]] = 224,
      patch_size: Union[int, Tuple[int, int]] = 16,
      in_chans: int = 3,
      embed_dim: int = 768,

3. dinov2/data/augmentations.py
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):

4. dinov2/configs/ssl_default_config.yaml     ###########################
  local_crops_scale:
  - 0.05
  - 0.32
  global_crops_size: 224
  local_crops_size: 96
evaluation:
  eval_period_iterations: 12500



# run time error, B*2 iter is halfed! then error
Data Loader Configuration:

Ensure that the DataLoader is not using persistent_workers=True unless necessary.
Set pin_memory=True only if you observe benefits, as it can sometimes increase memory usage.
Data Caching:

If you've implemented any caching mechanism in your dataset or data loader, verify that the cache size is bounded and old data is properly evicted.
Data Augmentation:

Complex data augmentation operations may inadvertently hold onto resources.
Check if any augmentation transforms are storing state that accumulates over time.


