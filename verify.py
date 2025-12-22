import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("ROCm-enabled PyTorch build:", hasattr(torch.version, 'rocm'))

    if torch.cuda.is_available():
        print("PyTorch can access the GPU.")
        print("Device name:", torch.cuda.get_device_name(0))
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print("GPU properties:")
        print(torch.cuda.get_device_properties(0))
    else:
        print("PyTorch cannot access the GPU. Please check your ROCm installation and Docker setup.")

if __name__ == "__main__":
    main()
