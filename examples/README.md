# R Version:

https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-18-04-quickstart
R 4.2.1


<!-- `wget https://cran.r-project.org/src/base/R-4/R-4.2.1.tar.gz`

`tar xvf R-4.2.1.tar.gz`

`cd R-4.2.1`

`./configure --prefix=/share/apps/R-4.2.1 --enable-R-shlib`

`make`

`make install` -->

`wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh`

`bash Anaconda3-2023.03-1-Linux-x86_64.sh`

`source .bashrc`

`conda config --add channels conda-forge`

`conda config --set channel_priority strict`

`conda create -n pyenv37 python=3.7`

`conda activate pyenv37`

`conda install -c conda-forge r-base=4.2.1`

`wget https://us.download.nvidia.com/tesla/460.106.00/NVIDIA-Linux-x86_64-460.106.00.run`

`sudo bash NVIDIA-Linux-x86_64-460.106.00.run`


# Packages

## CAM
`install.packages("remotes")`

`library(remotes)`

`install_github("cran/CAM")`

## pcalg

`install.packages("BiocManager")`

`BiocManager::install(c("graph", "RBGL", "ggm", "fastICA"))`

`install.packages("pcalg")`


## run
`sudo yum install tmux`

`tmux`

`pip install -r requirements.txt`

"Run job"

`crt + b`

`d`

 export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
