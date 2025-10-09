# RoCK-KV

### Get the code:
```
git clone https://github.com/Summer-Summer/RoCK-KV.git
cd RoCK-KV
git submodule update --init --recursive
```

### Building .sif image for apptainer:
```
mkdir build
cd build
sudo apptainer build kchanboost.sif ../rock_kv_cuda121.def 
```

### Building .img (writable overlay image):
```
cd build
apptainer overlay create --size 8192 kchanboost.img
```

### Installing software into the overlay image:
Entering the apptainer:
```
apptainer exec --nv \
--bind /home/$USER:/workspace \
--overlay build/kchanboost.img build/kchanboost.sif bash
```

Installing the package:
```
# 安装自定义transformers
cd /workspace/RoCK-KV/third_party/transformers
pip install -e .
# 安装自定义lm-evaluation-harness
cd /workspace/RoCK-KV/third_party/lm-evaluation-harness
pip install -e .
# 安装lm-eval with math support
pip install "lm-eval[math]"
# 安装KChanBoost
cd /workspace/RoCK-KV/
pip install -e .
```


### Exit the Apptainer.
```
exit
```


### Testing:
Entering the computing node (interative mode):
```
srun --ntasks=1 \
		 --gres=gpu:1 \
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
--overlay build/kchanboost.img build/kchanboost.sif bash
```

Testing:
```
cd /workspace/RoCK-KV/tests/
./gen_test.sh
```

```
srun --nodelist=research-external-03 --pty bash -i
```