conda create -y --prefix clam python=3.9
conda activate clam
pip install -e .


# the order of installing torch and jax matters
# install torch, make sure to install the correct version for your cuda
pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# might need to install 
# pip3 install nvidia-cudnn-cu12-8.9.2.26

# might also need to run this if there is a ptxas related issue
# https://github.com/google/jax/discussions/10327
# conda install cuda -c nvidia
