# neural-radiance-fields
<p align="center">
  <video src="https://github.com/user-attachments/assets/53a4abd4-f6b3-4479-9496-bbe187822dfe"
         width="600"
         autoplay
         loop
         muted
         playsinline>
  </video>
</p>
A minimal, from-scratch implementation of NeRF for volumetric rendering and view synthesis.

> **Note:**
> The wave-like distortion in the LEGO rotation comes from using **fixed-size raymarching steps** in the analytic renderer.

### **installation**
First install **PyTorch** appropriate for your system. Check your CUDA version with `nvidia-smi`, then:

```bash
# for cuda 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# for cuda 12.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest of the dependencies:
```bash
uv sync && source .venv/bin/activate
```

### **dataset preparation**
This project uses the NeRF synthetic dataset from:
🔗 [https://www.matthewtancik.com/nerf](https://www.matthewtancik.com/nerf)

Place the scenes inside the `dataset/` directory, e.g.:

```
dataset/
└── lego/
    ├── images/
    ├── transforms_train.json
    ├── transforms_test.json
    └── transforms_val.json
```

`NeRFDataset` expects the standard `transforms_*.json` structure.

**train NeRF**

```bash
python train.py
```

**Evaluate a trained NeRF**

```bash
python evaluate.py
```
Outputs include PSNR metrics and rendered trajectories

### **references**
- Mildenhall, Ben, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng.
  **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**.
  *European Conference on Computer Vision (ECCV)*, 2020.
  [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
