# neural-radiance-fields

dataset: https://www.matthewtancik.com/nerf

### **installation**
First install **PyTorch** appropriate for your system. Check your CUDA version with `nvidia-smi`, then:

```bash
# for cuda 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# for cuda 12.8
uv pip install torch torchvision
```

Then install the rest of the dependencies:
```bash
uv sync && source .venv/bin/activate
```

### **references**
[ ]
