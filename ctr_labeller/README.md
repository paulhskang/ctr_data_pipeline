
# Install

Install SAM,
https://github.com/facebookresearch/segment-anything
using their installation instructions with CONDA, and CUDA. Make sure to have a GPU with sufficient memory.

Download the model by putting in your browser: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`, then copy
paste model here. 


For configuring yaml code taken from kaolin-wisp
```bash
conda activate sam
pip install docstring_parser hydra-zen tyro
```

Install pandas
```bash
pip install pandas
```

# Download Dataset

Automatic download doesn't work for now. Download manually and put in structure:
```
.
├── data
|   |── ctr_capture_apr_25_24
|       |- cam1_0001.png
|       |- cam1_0002.png
|       |- ...      .png
|
|- sam_vit_h_4b839.pth
```

# Run

```bash
python3 main_ui.py --config config/config.yaml
```
