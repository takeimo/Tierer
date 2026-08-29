# 👑 Tierer

[![Python 3.10 - 3.13](https://img.shields.io/badge/python-3.10%20--%203.13-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast%20package%20manager-2BA8E2)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Tierer** is a modernized revival of the seminal layered depth inpainting project, [3D Photography using Context-aware Layered Depth Inpainting (CVPR 2020)](https://github.com/vt-vl-lab/3d-photo-inpainting).

The original project was a groundbreaking milestone in synthesized parallax generation from a single RGB image. **Tierer** updates this classic technology for modern operating systems and modern Python ecosystems using **[uv](https://github.com/astral-sh/uv)** — allowing anyone to run the full pipeline out of the box with a single command.

---

## ✨ Key Highlights

* **⚡ Single-Command Execution via `uv`**: Powered entirely by `uv`, which instantly provisions the exact Python environment and locked dependencies without manual environment setup.
* **🐍 Modern Ecosystem Compatibility**: Fully refactored and tested for **Python 3.10–3.13**, **PyTorch 2.6+**, and **NumPy 2.x**. Modernized for recent library updates (NumPy 2.x type standards, NetworkX 3.x graph APIs, safe PyYAML loading, and updated VisPy/MoviePy backends).
* **🤖 Automatic Model Download**: No manual downloads or file placements required. Pretrained weights are automatically fetched and set up on the first run.
* **💻 Broad Hardware Support**: Validated on Windows 11 on both discrete NVIDIA GPUs (e.g., RTX 3060) and entry-level CPUs (e.g., Intel N150) without local C++ compilation.

---

## 🚀 Quick Start

### Prerequisites
* **`git`** and **`uv`** are required. This guide assumes both are already installed on your system.

*(Note: Google Colab and cloud notebook environments are not supported. This project is dedicated to local execution via `uv`.)*

### Clone & Run
Clone the repository and run the pipeline. `uv` will automatically set up Python, install all locked dependencies, download required models, and process images:

```bash
git clone https://github.com/takeimo/Tierer.git
cd Tierer

# Run the pipeline (uv sets up and executes everything automatically)
uv run main.py --config argument.yml
```

### Custom Images & GPU Settings
* **Input / Output**: Place your input images into the `image/` directory. Output 3D mesh files (`mesh/*.ply`) and rendered parallax videos (`video/*.mp4`) will be generated automatically.
* **GPU / CPU Acceleration**: Edit `argument.yml` to select your compute device:
  * `gpu_ids: 0` — for NVIDIA GPU acceleration (CUDA)
  * `gpu_ids: -1` — for CPU-only execution

---

## 🧭 Project Policy & Scope

* **Single Responsibility**: The purpose of this repository is strictly to provide a stable, out-of-the-box implementation of **Layered Depth Inpainting (LDI)** in modern environments.
* **No Feature Creep**: We do not plan to add complex camera controls, UI wrappers, or new editing features. The pipeline outputs clean `.ply` 3D meshes so users can freely import and animate them in 3D software (such as Blender).
* **Continuous Modernization**: We prioritize tracking modern Python and library updates, keeping the pipeline operational on contemporary systems.
* **Calendar Versioning (`YYYY.MM.DD`)**: Releases and tags use calendar dates so users can easily see when the repository was last updated and verified.
* **Local-First Focus**: Support is limited to local CLI environments managed via `uv`. Google Colab / Jupyter notebooks are explicitly outside the scope of this project.

---

## ⚠ Disclaimer & Maintenance Notice

* **AI-Assisted Maintenance**: Please note that the maintainer is **not a professional software engineer**. This modernization project is curated and maintained with the assistance of AI/LLM tools.
* **No Pull Requests & Limited Support**: Because the maintainer lacks the expertise to review and validate external code, **pull requests are not accepted, and bespoke user support cannot be provided.** If you encounter platform-specific issues or wish to adapt the code, please feel free to fork this repository under the MIT License.

---

## 🙏 Heartfelt Acknowledgments & Citations

We express our deepest gratitude and respect to the original authors and researchers whose ingenious work made single-image layered depth inpainting possible:

* **Original Paper & Framework**:
  > **3D Photography using Context-aware Layered Depth Inpainting**  
  > Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang  
  > *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020*  
  > [[Paper](https://arxiv.org/abs/2004.04727)] | [[Project Page](https://shihmengli.github.io/3D-Photo-Inpainting/)] | [[Original Repository](https://github.com/vt-vl-lab/3d-photo-inpainting)]

```bibtex
@inproceedings{shih20203dphoto,
  title={3D Photography using Context-aware Layered Depth Inpainting},
  author={Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

* **Underlying Core Technologies**:
  * **MiDaS** (Monocular Depth Estimation): René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun ([isl-org/MiDaS](https://github.com/isl-org/MiDaS))
  * **EdgeConnect** (Adversarial Edge Inpainting): Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, Mehran Ebrahimi ([knazeri/edge-connect](https://github.com/knazeri/edge-connect))
  * **Partial Convolution** (Inpainting for Irregular Holes): Guilin Liu et al., NVIDIA ([ECCV 2018](https://arxiv.org/abs/1804.07723))

---

## 📄 License & Model Usage Disclaimers

* **Source Code**: Released under the [MIT License](LICENSE), consistent with the original repository.
* **Pretrained Weights**: The model checkpoints automatically downloaded from Hugging Face are derived from the original research and trained on academic datasets (such as MegaDepth). Users are responsible for ensuring compliance with the respective research and dataset licenses when using these models.
