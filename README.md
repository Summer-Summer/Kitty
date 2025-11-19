# RoCK-KV

### Get the code:
```
git clone -b jisen_dev https://github.com/Summer-Summer/RoCK-KV.git
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

**注意事项：** 在安装之前，有可能需要手动切换到以下分支：
- Transformers: `hf-4.53.2`
- lm_eval: `rock-kv`

```
# 安装自定义transformers
cd /workspace/RoCK-KV/third_party/transformers
git checkout hf-4.53.2  # 切换到指定分支
pip install -e .
# 安装自定义lm-evaluation-harness
cd /workspace/RoCK-KV/third_party/lm-evaluation-harness
git checkout rock-kv  # 切换到指定分支
pip install -e .
# 安装lm-eval with math support
pip install "lm-eval[math]"
# 安装KChanBoost
cd /workspace/RoCK-KV/
pip install -e .
# 安装seaborn用于数据可视化
pip install seaborn
# 安装HQQ用于HuggingFace原生KV Cache量化
pip install hqq
```


### Exit the Apptainer.
```
exit
```

### Testing:
Just testing, not for formal experiments.

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
--overlay build/kchanboost.img build/kchanboost.sif bash
```

### Running Evaluations:
```bash
cd eval_scripts

# View usage and parameters
./accuracy_eval5.sh

# Examples:
# Run with 10 repeats, batch_size=2
./accuracy_eval5.sh "Qwen/Qwen3-8B" "aime24" "0" "10" "2"

# Run with defaults (1 repeat, batch_size=1)
./accuracy_eval5.sh "Qwen/Qwen3-8B" "aime24" "0"

# Multi-GPU evaluation
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime25" "0,1" "10" "1"
```
cd /workspace/RoCK-KV/tests/
./gen_test.sh
```

```
srun --nodelist=research-external-03 --pty bash -i
```
