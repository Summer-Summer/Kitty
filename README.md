# RoCK-KV

Docker on Donglin
```
docker run -it --gpus all \
--shm-size=4gb \
--cap-add=CAP_SYS_ADMIN \
--runtime=nvidia \
--name PyTorch24_Haojun \
-v /scratch/workspaces/Haojun/:/home/xhjustc/ \
-v /data/:/data/ \
nvcr.io/nvidia/pytorch:24.04-py3 \
bash
```

Docker on Xiaoxia
```
docker run -it --gpus all \
--shm-size=4gb \
--cap-add=CAP_SYS_ADMIN \
--runtime=nvidia \
--name PyTorch24_Haojun \
-v /scratch/workspaces/xhjustc/:/home/xhjustc/ \
-v /data/:/data/ \
nvcr.io/nvidia/pytorch:24.04-py3 \
bash
```