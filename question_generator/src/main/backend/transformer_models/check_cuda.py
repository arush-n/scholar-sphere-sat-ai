# check_cuda.py
import torch, sys
print("Python exec:", sys.executable)
print("torch version:", torch.__version__)
print("CUDA toolkit:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name  :", torch.cuda.get_device_name(0))
