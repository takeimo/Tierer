# 👑 Tierer

[![Python 3.10 - 3.13](https://img.shields.io/badge/python-3.10%20--%203.13-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast%20package%20manager-2BA8E2)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Tierer** is a modernized revival of the seminal layered depth inpainting project, [3D Photography using Context-aware Layered Depth Inpainting (CVPR 2020)](https://github.com/vt-vl-lab/3d-photo-inpainting).

The original project was a groundbreaking milestone in synthesizing 3D parallax and context-aware disocclusion from a single RGB image. **Tierer** refactors this classic technology for contemporary systems using **[uv](https://github.com/astral-sh/uv)** — providing a zero-setup, fully isolated, and deterministic pipeline out of the box with a single command.

---

## ✨ Key Highlights

* **⚡ Single-Command Execution via `uv`**: Fully managed by `uv`. Automatically provisions an isolated local virtual environment (`.venv`) and pre-built wheels without polluting your global system or breaking other AI environments.
* **🐍 Modern Ecosystem Compatibility**: Refactored for **Python 3.10–3.13**, **PyTorch 2.6+**, and **NumPy 2.x**. Resolved legacy incompatibilities (NumPy 2.x type standards, NetworkX 3.x graph APIs, safe PyYAML loading, and Qt6/MoviePy backends).
* **🤖 Automatic Model Download**: Pretrained neural network weights are automatically fetched and verified from [Hugging Face](https://huggingface.co/takeimo/tierer-models) on the initial run.
* **🧠 Automatic Hardware Detection**: Automatically detects NVIDIA GPUs (CUDA) for high-speed processing (e.g., ~3 minutes on RTX 3060) or seamlessly falls back to CPU mode on machines without dedicated GPUs. No configuration needed.
* **🖼 Multi-Format Image Support**: Out-of-the-box support for **PNG (including RGBA/transparency)**, **JPEG/JPG**, **WebP**, and **BMP**.

---

## 🚀 Quick Start

### Prerequisites
* **`git`** and **`uv`** are required. (Ensure both are installed on your system).

*(Note: Google Colab and cloud notebooks are unsupported. Tierer is strictly tailored for deterministic local CLI execution via `uv`.)*

### 1. Clone & Run
Clone the repository and run the pipeline. `uv` will automatically set up the isolated environment, fetch locked dependencies and AI models, and render the demo:

```bash
git clone https://github.com/takeimo/Tierer.git
cd Tierer

# Run the pipeline (everything is handled automatically)
uv run main.py --config argument.yml
```

### 2. Processing Your Own Images
1. Drop your target image(s) (`.png`, `.jpg`, `.webp`, etc.) into the **`input_images/`** folder.
2. Run `uv run main.py --config argument.yml`.
3. Generated results are cleanly organized in **`outputs/`**:
   * 🎥 **Parallax Videos**: `outputs/videos/<image_name>_<trajectory>.mp4`
   * 🧊 **3D Inpainted Meshes**: `outputs/3d_meshes/<image_name>.ply`
   * 🗺 **Depth Maps**: `outputs/depth_maps/<image_name>.png`

---

## 🧭 Project Policy & Scope

* **Single Responsibility**: The sole purpose of this repository is to provide a rock-solid, out-of-the-box baseline for **Layered Depth Inpainting (LDI)** in modern environments.
* **No Feature Creep**: There are no plans to add complex camera GUIs, web interfaces, or interactive mesh editing. The pipeline exports clean `.ply` 3D meshes so users can freely import and animate them in 3D suites like Blender.
* **Continuous Modernization**: The focus remains on keeping `uv.lock` aligned with contemporary Python and upstream library releases to prevent codebase bitrot.
* **Calendar Versioning (`YYYY.MM.DD`)**: Releases and Git tags utilize calendar dates, making it immediately transparent when the repository was last verified and maintained.
* **Local-First Focus**: Exclusively optimized for local command-line execution.

---

## ⚠ Disclaimer & Maintenance Notice

* **AI-Assisted Maintenance**: The maintainer is **not a professional software engineer**. This modernization project is curated and maintained with the assistance of AI/LLM tools.
* **No Pull Requests & Limited Support**: Because the maintainer lacks the expertise to review external code, **pull requests are not accepted, and bespoke user support cannot be provided.** If you encounter platform-specific issues or wish to modify the code, please feel free to fork this repository under the MIT License.

---

## ⭐ Support the Project

If you found **Tierer** useful or enjoyed seeing your 2D photos come to life with 3D depth, **please consider giving this repository a star!** ⭐  

---

## 🙏 Heartfelt Acknowledgments & Citations

Deep gratitude and respect to the original authors and researchers whose ingenious work made single-image layered depth inpainting possible:

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

* **Source Code**: Released under the [MIT License](LICENSE), inherited from the original repository.
* **Pretrained Weights**: The checkpoints automatically fetched from Hugging Face are derived from the original research and trained on academic datasets (such as MegaDepth). Users are responsible for ensuring compliance with original research licenses when utilizing these models.
