### Raw data
data - https://github.com/haghish/openpsychometrics/blob/main/HEXACO
scoring key - https://ipip.ori.org/newHEXACO_PI_key.htm

### Dependencies

This project requires the following Python libraries:

-   `torch`
-   `pandas`
-   `transformers`
-   `datasets`
-   `peft`
-   `trl`
-   `bitsandbytes` (for 4-bit quantization in `finetune_custom.py`)
-   `unsloth` (for optimized training in `finetune_on_runpod.py`)

You can install the common dependencies using pip:
`pip install torch pandas transformers datasets peft trl`

For `finetune_custom.py`, also install `bitsandbytes`:
`pip install bitsandbytes`

For `finetune_on_runpod.py`, install `unsloth` as specified in its environment setup:
`pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
`pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes`

### Mistral 7B Fine-tuning with AMD GPU (Docker)

This section provides instructions for fine-tuning the Mistral 7B model using an AMD GPU via Docker on a Windows machine with WSL2.

#### Prerequisites:
1.  **WSL2 Installation:** Follow the official Microsoft guide to install and set up WSL2: [https://learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install)
2.  **AMD Drivers for WSL:** Install the necessary AMD drivers for WSL from: [https://www.amd.com/en/support/kb/release-notes/rn-rad-win-wsl-support](https://www.amd.com/en/support/kb/release-notes/rn-rad-win-wsl-support)
3.  **Docker Desktop:** Install Docker Desktop for Windows from: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
    *   Ensure "Use the WSL 2 based engine" is enabled in Docker Desktop **Settings > General**.
    *   Enable WSL integration for your installed distribution (e.g., Ubuntu) in **Settings > Resources > WSL Integration**.
    *   Restart Docker Desktop after applying changes.

#### Running the Docker Container:

1.  **Open a WSL terminal** (e.g., Ubuntu).
2.  Navigate to the project directory where `verify.py` and `finetune.py` are located.
3.  Run the following command to start the Docker container with GPU access and mount your current project directory:

    ```bash
    docker run -it --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v "$(pwd):/workspace" rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2 /bin/bash
    ```
    *   **Note:** If you are using the old Windows Command Prompt (cmd.exe) instead of PowerShell or a WSL terminal, replace `$(pwd)` with `%cd%`.

#### Inside the Docker Container:

Once inside the Docker container's bash shell, follow these steps:

1.  **Install Python Dependencies:**
    ```bash
    pip install transformers datasets accelerate peft trl bitsandbytes pandas
    ```

2.  **Verify GPU Access:**
    Run the verification script to ensure PyTorch can detect and utilize your AMD GPU:
    ```bash
    python verify.py
    ```
    You should see output indicating that PyTorch can access the GPU and list its properties.

3.  **Run the Fine-tuning Script:**
    After successful verification, you can run the fine-tuning script. Remember to adjust the `MODEL_NAME` and `DATASET_NAME` in `finetune.py` as needed for your specific use case.
    ```bash
    python finetune.py
    ```

The fine-tuned model and tokenizer will be saved in the `mistral_7b_finetuned/final_checkpoint` directory within your project workspace.
