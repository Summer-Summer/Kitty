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
--bind /data:/data \
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


### Updating packages in overlay (开发过程中更新代码):
如果修改了lm_eval或RoCK-KV的代码，需要重新安装到overlay中：

进入容器环境:
```
apptainer exec --nv \
--bind /home/$USER:/workspace \
--bind /data:/data \
--overlay build/kchanboost.img build/kchanboost.sif bash
```

更新对应的包:
```
# 如果修改了lm-evaluation-harness代码
cd /workspace/RoCK-KV/third_party/lm-evaluation-harness
pip install -e . --force-reinstall

# 如果修改了RoCK-KV代码
cd /workspace/RoCK-KV/
pip install -e . --force-reinstall

退出容器:
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
--bind /data:/data \
--overlay build/kchanboost.img build/kchanboost.sif bash
```

Testing:
```
cd /workspace/RoCK-KV/tests/
./gen_test.sh
```