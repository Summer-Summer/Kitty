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

**Note:** We no longer use interactive nodes for running evaluations. The following commands are provided as background knowledge only.

- **Request an interactive node:** 
  ```bash
  srun --ntasks=1 --gres=gpu:8 --cpus-per-task=64 --mem=1024000 --partition=batch --pty /bin/bash
  ```
  
  **Note:** The `--mem=1024000` requests 1TB of RAM. If you encounter RAM shortage issues, you can reduce memory usage by setting `low_cpu_mem_usage=True` in the model loading code (`src/rock_kv/cli/eval_rock_kv.py` line 34).

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

## Prerequisites

### 1. Set Up GitHub SSH Access

To push your code changes and results, you need to configure SSH authentication with GitHub.

**Quick setup:**
```bash
# Generate SSH key and add to GitHub
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # Copy this output to GitHub
```

Then add the key to GitHub: https://github.com/settings/keys

For detailed instructions, see: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### 2. Request Model Access

Before running evaluations, you need to request access to the following models/datasets on Hugging Face:

- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (for potential future use)
- [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)

Visit each link above, log in to your Hugging Face account, and click the "Request Access" button. Approval is usually granted within a few minutes to hours.

### 3. Set Up Hugging Face Token

After gaining access, you need to authenticate with your Hugging Face token:

**Option 1: Use CLI login**
```bash
huggingface-cli login
```
This will save your token to `~/.cache/huggingface/token` permanently.

**Option 2: Add to your shell profile**
```bash
# Add to ~/.bashrc for persistent authentication
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**Option 3: Temporary authentication (current session only)**
```bash
export HF_TOKEN="your_token_here"
```

You can find your token at: https://huggingface.co/settings/tokens

## Running Evaluations

1. Open `sbatch/run_eval_rupert.sh`

2. Modify the important parameters:
   - **A. mem** = 500G/1000G (for 70B models, you may need 1T)
   - **B. Runtime**: Estimate based on the online spreadsheet; for Qwen3-32B we use 250h
   - **C. --nodelist**: Check hostname availability using `sinfo` to find which host is idle
   - **D. MODEL, TASK_NAME, NUM_REPEATS, BATCH_SIZE**: Modify as needed
   - **E. Debug mode**: Set `DEBUG=1` to run only 3 questions for testing
   - **F. Experiment configuration**: Comment out experiments in `EXPERIMENTS` array to control how many experiments to run

3. Submit the job from the login node:
   ```bash
   sbatch ~/RoCK-KV/sbatch/run_eval_rupert.sh
   ```

