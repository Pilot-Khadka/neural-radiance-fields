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

dataset: https://www.matthewtancik.com/nerf

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

### **references**
- Mildenhall, Ben, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng.
  **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**.
  *European Conference on Computer Vision (ECCV)*, 2020.
  [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
