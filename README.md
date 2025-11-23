# Kitty
Kitty is a plug-and-play KV-cache system for HuggingFace Transformers, enabling accurate 2-bit KV-cache quantization through channel-wise precision boost.  
This repository is the official artifact of our conference submission (under review).

### Get the code:
```
git clone https://github.com/Summer-Summer/Kitty.git
cd Kitty
git submodule update --init --recursive
```

### Building .sif image for apptainer:
```
mkdir build
cd build
sudo apptainer build kitty.sif ../kitty_cuda121.def 
```

### Building .img (writable overlay image):
```
cd build
apptainer overlay create --size 8192 kitty.img
```

### Installing software into the overlay image:
Entering the apptainer:
```
apptainer exec --nv \
--bind /home/$USER:/workspace \
--overlay build/kitty.img build/kitty.sif bash
```

Installing the package:

**Important:** Before installation, we need to manually switch the submodules to the following branch：
- Transformers: `hf-4.53.2`
- lm_eval: `kitty`

```
# Install transformers
cd /workspace/Kitty/third_party/transformers
git checkout hf-4.53.2
pip install -e .
# Install lm-evaluation-harness
cd /workspace/Kitty/third_party/lm-evaluation-harness
git checkout kitty
pip install -e .
# Install lm-eval with math support
pip install "lm-eval[math]"
# Install Kitty
cd /workspace/Kitty/
pip install -e .
# Install seaborn for visualization
pip install seaborn
# Install HQQ for HuggingFace's KV Cache quantization
pip install hqq
```


### Exit the Apptainer.
```
exit
```

### Run experiments

#### Before Runing the experiments:
Entering the computing node (interative mode):
```
srun --ntasks=1 \
		 --gres=gpu:8 \
		 --cpus-per-task=64 \
		 --mem=450000 \
		 --partition=batch \
     --job-name=debug \
     --pty /bin/bash
```

Entering the apptainer:
```
apptainer exec --nv \
--bind /home/$USER:/workspace \
--overlay build/kitty.img build/kitty.sif bash
```

#### Runing latency banchmarking:
See more details in [latency_benchmarking](latency_benchmarking/).

#### Runing accuracy simulation:
See more details in [accuracy_simulation](accuracy_simulation/).


## Citation

If you find Kitty useful or relevant to your research, please kindly cite [our paper (to be added)]():

```

```