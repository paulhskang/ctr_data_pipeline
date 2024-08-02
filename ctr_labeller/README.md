
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

# Dataset and Reference csv file

Automatic download doesn't work for now.

Masks are created in the same directory as the reference file. Structure for reference file and newly created masks/images by SAM:
```
├── some_folder_on_computer
|   |── jul24                   # data for one set of tubes
|       |- reference.csv        # reference file
|       |- input_prompts        # created by this program
|       |- imgs                 # main folder for data
|           |── run1                # different folders per run as data collection could be done over several days
|               |- 0                # batch numbers to limit number of images per folder (currently 1000 pairs per)
|                   |- filename1_cam0.jpg
|                   |- filename1_cam1.jpg
|                   |- ...
|               |- 1
|                   |- filename1001_cam0.jpg
|                   |- filename1001_cam1.jpg
|                   |- ...
|               |- ...
|           |── run2
|               |- 10
|                   |- filename10001_cam0.jpg
|                   |- filename10001_cam1.jpg
|                   |- ...
|               |- 11
|                   |- filename11001_cam0.jpg
|                   |- filename11001_cam1.jpg
|                   |- ...
|               |- ...
|           |── ...
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
|       |- image_and_masks                # created by this program
```

One single reference csv file should hold all the relative paths to each stereoimage pair.

| frame_id      | Joints (12 total) | batch_num | left_image_path | right_image_path |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | ## | 0 | /imgs/run1/0/filename1_cam0.jpg | /imgs/run1/0/filename1_cam1.jpg |
| 2 | ## | 0 | /imgs/run1/0/filename2_cam0.jpg | /imgs/run1/0/filename2_cam1.jpg |

# Run
To run GUI program:
```bash
python main_ui.py --config config/config.yaml --data-path /path/to/reference/file
```

To run offline segmentation program:
```bash
python main_offline.py --config config/config.yaml --data-path /path/to/reference/file
```

# TODO:
- Multithreaded dataloader
- Diplay which frames on screen/in terminal?