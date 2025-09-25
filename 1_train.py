
import subprocess
#import numpy as np
import os
import logging
#a = np.array([1,2])
# Set up logging configuration (you may want to configure this according to your needs)
# Define the log file path
log_file_path = "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/logfile.log"
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to the console
        ])
logger = logging.getLogger(__name__)

def before():
    command = "pip freeze > /rsrch1/ip/msalehjahromi/before_requirements.txt"
    subprocess.run(command, shell=True, check=True)

def after():
    command = "pip freeze > /rsrch1/ip/msalehjahromi/after_requirements.txt"
    subprocess.run(command, shell=True, check=True)


def install_packages():
    commands = [
        ["pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath", "xformers==0.0.18", "submitit","numpy<2.0"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
        ["pip", "install", "mmcv-full==1.5.0"]
         ]
    for i,command in enumerate(commands):
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #np.save(f"/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/package{i}_installed_in_train.npy", a)

def run_training():
    #"--nodes", "1",
    #"--gpus" , "8",
    command = [
        "/usr/bin/python3", "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/dinov2/run/train/train_torchrun.py",
        "--config-file", "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/dinov2/configs/train/vitl16_short.yaml",
        "--output-dir", "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/output_dir/448_192_all_test",
        "train.dataset_path=CTDataset:npz_dir_path=/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Batch_1_3SlicesNpy600k-v2:patch_size=512,512,3"
    ]
    # Print the command to be executed for debugging
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"Training output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error:\n{e.stderr}")


if __name__ == "__main__":
    before()
    install_packages()
    after()
    import numpy as np
    from dinov2.data.datasets import ImageNet
    a = np.array([1,2])
    np.save("/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/zzz_all_packages_installed.npy", a)
    result = subprocess.run(["/usr/bin/python3", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    formatted_version = result.stdout.strip().replace(" ", "_").replace(".", "_")
    import torch
    n_gpus = str(torch.cuda.device_count())
    np.save(f"/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6/yamls/zzz_{n_gpus}_gpus.npy", a)
    os.environ['PYTHONPATH'] = "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6"
    run_training()
