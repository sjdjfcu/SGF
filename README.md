# EviNav

## Installation

Install the system dependencies:

```bash
sudo apt update
sudo apt install cmake build-essential \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libgl1-mesa-dev libegl1-mesa-dev libglfw3-dev libopencv-dev
```

Create and activate the Conda environment:

```bash
conda create -n sgf python=3.12
conda activate sgf
```

Clone the dependencies into your home/workspace directory. The commands below assume
the SGF repository is at `~/SGF`.

```bash
cd ~/
git clone https://github.com/facebookresearch/habitat-sim.git --recursive
git clone https://github.com/facebookresearch/habitat-lab.git
git clone https://github.com/verlab/accelerated_features.git
```

Install PyTorch, Habitat, and the remaining dependencies, then build the
SGF extension:

```bash
pip3 install torch torchvision

cd ~/habitat-sim/
pip install scikit-build-core pybind11
pip install . --no-build-isolation  # May take more than 30 minutes on some machines.

cd ~/habitat-lab/
pip install -e habitat-lab
pip install -e habitat-baselines

cd ~/SGF
pip install ultralytics
pip install git+https://github.com/ultralytics/CLIP.git
pip install scikit-image scikit-fmm
pip install numpy==1.26.4

python tools/build_extension.py
```

Habitat-Lab 0.3.3 requires `numpy==1.26.4`. Ignore the dependency
incompatibility warning reported after installing this NumPy version.

## Dataset

Download the following datasets:

- [HM3D scene dataset](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar).
- [Instance-image-goal navigation episodes](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip).
- [Text-goal navigation episodes](https://drive.google.com/uc?export=download&id=1KNdv6isX1FDZi4KCVPiECYDxijg9cZ3L).

Arrange the dataset as follows:

```text
SGF/
└── data/
    ├── datasets/
    │   ├── textnav/
    │   │   └── val/
    │   │       └── val_text.json.gz
    │   └── instance_imagenav/
    │       └── hm3d/
    │           └── v3/
    │               └── val/
    │                   ├── content/
    │                   │   ├── 4ok3usBNeis.json.gz
    │                   │   ├── 5cdEh9F2hJL.json.gz
    │                   │   ├── ...
    │                   │   └── zt1RVoi7PcG.json.gz
    │                   └── val.json.gz
    └── scene_datasets/
        └── hm3d_v0.2/
            └── val/
                ├── 00800-TEEsavR23oF/
                │   ├── TEEsavR23oF.basis.glb
                │   └── TEEsavR23oF.basis.navmesh
                ├── 00801-HaxA7YrQdEC/
                ├── ...
                └── 00899-58NLZxWBSpk/
```

If the data is stored elsewhere, link it from the SGF repository root.
Replace `/path/to/data` with the directory containing `datasets/` and
`scene_datasets/`:

```bash
ln -sfn "/path/to/data" data
```

## Multimodal Models

### Option 1: OpenAI-compatible API

Set the LLM/VLM model names and API endpoint base URL in
[python/sgf_habitat/configs/config_habitat.yaml](python/sgf_habitat/configs/config_habitat.yaml).

### Option 2: Local model

For example, serve Qwen3.5-4B with [vLLM](https://vllm.ai/):

```bash
vllm serve Qwen/Qwen3.5-4B \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

Configure the following fields in
[python/sgf_habitat/configs/config_habitat.yaml](python/sgf_habitat/configs/config_habitat.yaml):

```yaml
llm_model: Qwen/Qwen3.5-4B
vlm_model: Qwen/Qwen3.5-4B
api_key: ''
base_url: http://localhost:8000/v1
```

**Qwen3.5-2B:** When serving `Qwen/Qwen3.5-2B`, add
`--override-generation-config '{"temperature": 0.1}'` to the `vllm serve`
command. All LLM/VLM requests require JSON output. In testing, vLLM's default
temperature of `1.0` caused some episodes to receive invalid JSON, empty
responses, or incorrect JSON types. These issues were not observed with a
temperature of `0.1`.

## Evaluation

Run the commands below from the SGF repository root with the `sgf`
environment active.

```bash
export SGF_LLM_API_KEY="<API_KEY>"  # Optional; only needed if the server has an API key configured.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json  # Set if evaluation fails with LLVM ERROR: Option 'default' already exists!, especially on headless machines.

python scripts/init_nav_models.py
```

The initialization script downloads `yoloe-26s-seg.pt` and
`mobileclip2_b.ts` for text-goal navigation, and `RN50.pt` for
instance-image-goal navigation.

### Single episode

```bash
python scripts/run_habitat_text_image_goal_nav.py --task-mode text --episode-id 0 --verbose
python scripts/run_habitat_text_image_goal_nav.py --task-mode instance-image --episode-id 0 --verbose
```

Both task modes save outputs to
`outputs/sgf_text_instance_image_nav_runs/episode_<episode_id>/` by default,
relative to the repository root. The episode ID is zero-padded to four digits
(for example, `episode_0000`). Use `--artifact-root` to change the parent directory
or `--artifact-dir` to set the episode output directory directly.

### Batch evaluation

The following example uses two GPUs. Adjust `--num-workers` and
`--max-gpu-workers-per-device` to match your machine's capabilities.

```bash
python scripts/batch_eval_nav.py \
    --task-mode text \
    --episodes 0-999 \
    --max-steps 1000 \
    --timeout-seconds 1500 \
    --num-workers 12 \
    --max-gpu-workers-per-device 6

python scripts/batch_eval_nav.py \
    --task-mode instance-image \
    --episodes 0-999 \
    --max-steps 1000 \
    --timeout-seconds 1500 \
    --num-workers 12 \
    --max-gpu-workers-per-device 6
```

Default output roots, relative to the repository root:

| Task mode | Output root |
| --- | --- |
| `text` | `outputs/textnav_batch/` |
| `instance-image` | `outputs/instance_image_nav_batch/` |

Use `--output-root` to change the root.

Use `--runner-args "--metrics-only "` to skip episode logs and visualizations and keep evaluation
metrics.

### Instance-image evaluation without VLM

Single episode:

```bash
python scripts/run_habitat_text_image_goal_nav.py \
    --task-mode instance-image \
    --episode-id 0 \
    --nav-override image_vlm_stop_validation=false \
    --nav-override image_vlm_panoramic_query=false
```

Batch evaluation:

```bash
python scripts/batch_eval_nav.py \
    --task-mode instance-image \
    --episodes 0-999 \
    --max-steps 1000 \
    --timeout-seconds 1500 \
    --num-workers 12 \
    --max-gpu-workers-per-device 6 \
    --runner-args="--nav-override image_vlm_stop_validation=false --nav-override image_vlm_panoramic_query=false"
```

## Real Hardware (Kobuki Base + ZED 2i Stereo Camera)

### Kobuki driver installation

SGF uses the standalone C++ `kobuki_core` driver over USB; a ROS runtime is
not required. The source workspace follows the
[official standalone installation guide](https://kobuki.readthedocs.io/en/devel/software.html).
The commands below include the compatibility fixes used by the local
Ubuntu 26.04 / Python 3.14 / GCC 15 build and install into `~/kobuki/install`.

Install system dependencies on Ubuntu/Debian, then create a separate build
environment for the driver:

```bash
sudo apt update
sudo apt install -y build-essential cmake git wget python3-dev python3-venv

mkdir -p ~/kobuki/src
cd ~/kobuki
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip wheel
python3 -m pip install \
    setuptools==79.0.1 colcon-core==0.21.0 \
    colcon-common-extensions==0.3.0 vcstool==0.3.0 \
    catkin_pkg==1.1.0 empy==4.2.1

wget -O colcon.meta \
    https://raw.githubusercontent.com/kobuki-base/kobuki_documentation/release/1.0.x/resources/colcon.meta
wget -O kobuki_standalone.repos \
    https://raw.githubusercontent.com/kobuki-base/kobuki_documentation/release/1.0.x/resources/kobuki_standalone.repos
vcs import src < kobuki_standalone.repos
```

The repository manifest imports ament build tools, Eigen, Sophus, ECL, and
`kobuki_core` (`release/1.3.x`). Keep `colcon.meta` in the workspace root: it
identifies Eigen as a CMake package and disables Eigen/Sophus tests.
The pinned setuptools version supplies the legacy Python packaging interfaces
used by the imported ament tools.

With GCC 15, the imported ECL code needs the following compatibility fixes
before building. Run this from `~/kobuki`; it exposes the base error assignment
operators and replaces two static variable-length arrays:

```bash
python3 - <<'PY'
from pathlib import Path

for filename in (
    "src/ecl_lite/ecl_io/include/ecl/io/sockets.hpp",
    "src/ecl_lite/ecl_time_lite/include/ecl/time_lite/errors.hpp",
    "src/ecl_core/ecl_filesystem/include/ecl/filesystem/errors.hpp",
):
    path = Path(filename)
    text = path.read_text()
    if "using Error::operator=;" not in text:
        path.write_text(text.replace("public:", "public:\n  using Error::operator=;", 1))

path = Path("src/ecl_lite/ecl_sigslots_lite/include/ecl/sigslots_lite/managers.hpp")
text = path.read_text()
for data_type in ("Data", "void"):
    text = text.replace(
        f"static sigslots::GlobalSlot<{data_type}> slots[capacity];",
        f"static sigslots::GlobalSlot<{data_type}> *slots = new sigslots::GlobalSlot<{data_type}>[capacity];",
    )
path.write_text(text)
PY
```

Build the driver and all its dependencies into one install prefix.
`--merge-install` is required by SGF's bridge, which expects headers in
`install/include` and shared libraries in `install/lib`.
The [CMake policy option](https://cmake.org/cmake/help/latest/variable/CMAKE_POLICY_VERSION_MINIMUM.html)
allows CMake 4 to configure the older dependencies:

```bash
cd ~/kobuki
source .venv/bin/activate
colcon build --merge-install --cmake-args \
    -DBUILD_TESTING=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    --no-warn-unused-cli
deactivate
source ~/kobuki/install/setup.bash
```

Install the USB device rule used by this project. It creates `/dev/kobuki`,
grants serial access to the `dialout` group, and prevents ModemManager from
probing the base:

```bash
sudo tee /etc/udev/rules.d/60-kobuki.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="kobuki*", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", MODE:="0660", GROUP:="dialout", SYMLINK+="kobuki"
EOF
sudo usermod -aG dialout "$USER"
sudo udevadm control --reload-rules
```

Log out and back in so the group membership takes effect, then reconnect the
USB cable and turn on the base. Verify the device and driver connection:

```bash
source ~/kobuki/install/setup.bash
ls -l /dev/kobuki
test -r /dev/kobuki && test -w /dev/kobuki
kobuki-version-info
```

`kobuki-version-info` should report hardware, firmware, and software versions.
If `/dev/kobuki` is missing, check power, USB connection, and the udev rule;
if access is denied, check `id -nG` for `dialout` after logging in again.
Return to the SGF repository root and build the bridge as described below.
For a different install prefix, pass the same path to `--kobuki-install` when
building the bridge and when launching real navigation.

### ZED SDK and Kobuki bridge

Install the [ZED SDK](https://www.stereolabs.com/developers/release).
Then return to the SGF repository root, activate the `sgf` environment,
and build the bridge:

```bash
cd ~/SGF
conda activate sgf
source ~/kobuki/install/setup.bash
python3 tools/build_kobuki_bridge.py --kobuki-install ~/kobuki/install
```

### Camera calibration

Calibrate horizontal offsets and mounting yaw using live robot motion:

```bash
python3 scripts/calibrate_zed_base_online.py \
    --kobuki-device /dev/kobuki \
    --kobuki-install ~/kobuki/install \
    --zed-resolution HD720 \
    --zed-camera-base-y 0.5 \
    --output outputs/real/zed_base_calibration.json
```

### Text-goal navigation

```bash
python scripts/run_habitat_text_image_goal_nav.py \
    --real-env \
    --task-mode text \
    --goal-text "Find the tall beige trash bin with a red lid, red frame, and red foot pedal. A black bin liner hangs over its rim. It stands against a cream-colored wall in a blue-carpeted hallway, immediately to the left of a similar bin with a green lid and frame, near a blue door." \
    --kobuki-device /dev/kobuki \
    --kobuki-install ~/kobuki/install \
    --zed-resolution HD720 \
    --zed-output-width 1280 \
    --zed-output-height 720 \
    --zed-camera-base-y 0.5 \
    --max-steps 500 \
    --artifact-dir outputs/real/bin_text \
    --zed-extrinsics-path outputs/real/zed_base_calibration.json
```

### Instance-image-goal navigation

```bash
python scripts/run_habitat_text_image_goal_nav.py \
    --real-env \
    --task-mode instance-image \
    --goal-image-path bin.jpg \
    --kobuki-device /dev/kobuki \
    --kobuki-install ~/kobuki/install \
    --zed-resolution HD720 \
    --zed-output-width 1280 \
    --zed-output-height 720 \
    --zed-camera-base-y 0.5 \
    --max-steps 500 \
    --artifact-dir outputs/real/bin_img \
    --zed-extrinsics-path outputs/real/zed_base_calibration.json
```

## Acknowledgments and Citation

Built on top of an incremental [SceneGraphFusion (SGF)](https://github.com/ShunChengWu/SceneGraphFusion) backbone.
Please cite the following works:

```bibtex
@inproceedings{wu2021scenegraphfusion,
  title={Scenegraphfusion: Incremental 3d scene graph prediction from rgb-d sequences},
  author={Wu, Shun-Cheng and Wald, Johanna and Tateno, Keisuke and Navab, Nassir and Tombari, Federico},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={7511--7521},
  year={2021},
  organization={IEEE}
}

@inproceedings{tateno2015real,
  title={Real-time and scalable incremental segmentation on dense slam},
  author={Tateno, Keisuke and Tombari, Federico and Navab, Nassir},
  booktitle={2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4465--4472},
  year={2015},
  organization={IEEE}
}
```

## License

This project is licensed under the BSD 2-Clause License. See [LICENSE](LICENSE)
for the full terms and copyright notice.
