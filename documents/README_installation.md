# LSL Installation
In this section, we demonstrate how to prepare an environment with lsl_tools.
lsl_tools works on Linux Ubuntu 22.04 and Windows.
It requires Python3.8, CUDA 11.x, Pytorch 1.9.1 and GCC 8 or 9 (for Linux) or Microsoft C++ build tools 2019 (for Windows).

## Step0. Create a conda environment and activate it

```
conda create --name lsl_tools python=3.8
conda activate lsl_tools
```

## Step1. Install Pytorch

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Step2. Install building tools and required packages

### Linux-specific Installation
Please use following command for installation on Linux environment.
```
# install system packages of building tools
sudo apt-get install build-essential linux-generic libmpich-dev libopenmpi-dev
```

### Windows-specific Installation
For Windows environment, please make sure following packages have been installed
1. Microsoft C++ build tools 2019   
https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. CUDA Toolkit for windows 11.x   
https://developer.nvidia.com/cuda-toolkit-archive

And set necessary environment variables after build tools and CUDA Toolkit get installed.
```
set TORCH_CUDA_ARCH_LIST=5.2 6.1 7.0 7.5 8.0 8.6
set DISTUTILS_USE_SDK=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
```
*. The remaining steps of LSL Windows installation should be done in the same shell contains above environment variables.

### Installation python packages
Install the LSL dependency packages after above OS-specific operations are completed.
```
pip install -r requirements.txt
```

## Step3. Install LSL_Tools
```
git clone ${lsl_tools_git_url}
cd lsl-tools
git checkout lsl_0.9_oss
python setup.py develop

```

## Step4. Install third-part Packages
```
# Install Glip
cd ${new_work_space}
# ${new_work_space} can be set inside LSL_Tools or outside, but for the organization of git project, it's recommended to be set outside.

git clone https://github.com/microsoft/GLIP.git
cd GLIP
git reset --hard 81207d3a6ed1c27d31a5e62625c288ea7353e448
git apply ${lsl_tools_top_dir}/lsl_tools/glip_lsl/glip_lsl.patch
python setup.py develop

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install LabelStudio ML
cd ${lsl_tools_top_dir}/lsl_tools/labelstudio/label-studio-ml-backend
python setup.py develop
```



