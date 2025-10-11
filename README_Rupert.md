# RoCK-KV Evaluation Guide for Rupert

Hi Rupert, thank you for participating in this experiment.

## Setup

First, you need to configure the environment according to the instructions in [README_Slurm.md](README_Slurm.md).

After completing the setup, you should have a `build` folder containing:
- One `.sif` file (Singularity/Apptainer image)
- One `.img` file (writable overlay image)

We use Singularity as the container system. During the experimental phase, files that may be modified are stored in the `.img` overlay, which is mounted on top of the base `.sif` image.

## Useful Slurm Commands

If you are also using a Slurm-managed cluster, these commands will be helpful:

- **Request an interactive node:** 
  ```bash
  srun --ntasks=1 --gres=gpu:8 --cpus-per-task=64 --mem=450000 --partition=batch --pty /bin/bash
  ```

- **Enter a specific node from login node:**
  ```bash
  srun --nodelist=research-external-xx --pty bash -i
  ```

- **Check available resources:**
  ```bash
  sinfo
  ```

- **View your current jobs:**
  ```bash
  squeue -u $USER
  ```

- **Check GPU usage and process IDs:**
  ```bash
  nvidia-smi
  ```

## Running Evaluations

After requesting a compute node, you need to:

1. Enter the Apptainer container:
   ```bash
   apptainer exec --nv \
   --bind /home/$USER:/workspace \
   --bind /data:/data \
   --overlay build/kchanboost.img build/kchanboost.sif bash
   ```

2. Navigate to the evaluation scripts folder:
   ```bash
   cd /workspace/RoCK-KV/eval_scripts
   ```

3. Run experiments with Qwen3-32B:
   
   You may need to modify `accuracy_eval5.sh` and use terminal commands to run different experimental configurations in batches.

## Experimental Tasks

### 1. Baseline (FP16)

First, edit `accuracy_eval5.sh`:
- **Uncomment** the `run_hf_baseline` line
- **Comment out** all `run_single_exp` lines
- Save the changes

Run the following commands:
```bash
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime24" "0" "10" "1"
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime25" "1" "10" "1"
```

**Note:** After submitting any shell script, you can safely press `Ctrl + C` to exit - the task will continue running in the background. Logs can be found in `eval_scripts/eval_logs`, and results will be saved in `eval_scripts/eval_results`.

### 2. KIVI - K4V2

Edit `accuracy_eval5.sh`:
- **Comment out** `run_hf_baseline`
- **Uncomment** one `run_single_exp` line (the third one, but any will work as long as the parameters are correct)
- Set the parameters: `sink = 0`, `kbits = 4`, `vbits = 2`, `promote_ratio = 0.0`
- Keep other parameters unchanged
- Save the changes

Run the following commands:
```bash
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime24" "2" "10" "1"
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime25" "3" "10" "1"
```

### 3. KIVI - K2V2

Edit `accuracy_eval5.sh`:
- For `run_single_exp`: set `sink = 0`, `kbits = 2`, `vbits = 2`, `promote_ratio = 0.0`
- Keep other parameters unchanged
- Save the changes

Run the following commands:
```bash
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime24" "4" "10" "1"
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime25" "5" "10" "1"
```

### 4. KChanBoost-K2.1V2 (10% Promotion)

Edit `accuracy_eval5.sh`:
- For `run_single_exp`: set `sink = 32`, `kbits = 2`, `vbits = 2`, `promote_ratio = 0.1`
- Keep other parameters unchanged
- Save the changes

Run the following commands:
```bash
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime24" "6" "10" "1"
./accuracy_eval5.sh "Qwen/Qwen3-32B" "aime25" "7" "10" "1"
```

## Task Tracking

My habit is to record each submitted task in our online Excel spreadsheet using the format: `IP-GPU_ID`

## Advanced Tips: Running Multiple machines in Parallel

If you want to run 2-3 machines simultaneously, here's a method:

1. Copy the overlay image:
   ```bash
   cp build/kchanboost.img build/kchanboost2.img
   ```

2. When entering the container, mount the new image:
   ```bash
   apptainer exec --nv \
   --bind /home/$USER:/workspace \
   --bind /data:/data \
   --overlay build/kchanboost2.img build/kchanboost.sif bash
   ```

3. Follow the same steps as before.

**Important:** Do not let two nodes execute the same task, otherwise the logs may be overwritten.

