
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

Automatic download doesn't work for now.

Structure for reference file and newly created masks/images by SAM:
```
.
├── data
|   |── ctr_jul_26
|       |- reference.csv
|       |- input_prompts        # created by this program
|       |- masks                # created by this program
|           |- 0
|               |- mask_filename1_cam0.jpg
|               |- mask_filename1_cam1.jpg
|               |- ...
|           |- 1
|               |- mask_filename1001_cam0.jpg
|               |- mask_filename1001_cam1.jpg
|               |- ...
|           |- ...
|
|- sam_vit_h_4b839.pth
```

To prevent re-downloading and syncing issues, stereoimage data can be kept anywhere on the system. Structure:
```
├── some_folder_on_computer
|   |── ctr_jul_26              # data of one set of tubes
|       |── run1                # different folders per run as data collection could be done over several days
|           |- 0                # batch numbers to limit number of images per folder (1000 pairs per)
|               |- filename1_cam0.jpg
|               |- filename1_cam1.jpg
|               |- ...
|           |- 1
|               |- filename1001_cam0.jpg
|               |- filename1001_cam1.jpg
|               |- ...
|           |- ...
|       |── run2
|           |- 10
|               |- filename10001_cam0.jpg
|               |- filename10001_cam1.jpg
|               |- ...
|           |- 11
|               |- filename11001_cam0.jpg
|               |- filename11001_cam1.jpg
|               |- ...
|           |- ...
|       |── ...
```

# Reference csv
One single reference csv file should hold all the relative paths to each image.

| frame_id      | Joints (12 total) | batch_num | left_image_path | right_image_path |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | ## | 0 | /ctr_jul_26/run1/0/filename1_cam0.jpg | /ctr_jul_26/run1/0/filename1_cam1.jpg |
| 2 | ## | 0 | /ctr_jul_26/run1/0/filename2_cam0.jpg | /ctr_jul_26/run1/0/filename2_cam1.jpg |

# Run

```bash
python3 main_ui.py --config config/config.yaml --data-path data/ctr_capture_apr_25_24
```

# TODO:
- Picture zooming
- Multithreaded dataloader

- Picture/window resizing in GUI
- Diplay which frames on screen/in terminal?

- save mask with folder structure
- retrieve based on inline argument and csv filepath
e.g.,
```bash
python3 main_ui.py --config config/config.yaml --data-path data/ctr_jul_26 --image-dir ~/path/to/some_folder_on_computer
```