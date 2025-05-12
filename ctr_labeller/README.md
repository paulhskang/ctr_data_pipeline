
# Install

Install SAM,
https://github.com/facebookresearch/segment-anything
using their installation instructions with CONDA, and CUDA. Make sure to have a GPU with sufficient memory.

Download the model by putting in your browser: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`, then copy
paste model here. 


For configuring yaml code taken from kaolin-wisp
```bash
conda activate sam
pip install docstring_parser hydra-zen tyro pandas
```

# Dataset and Reference csv file

Automatic download doesn't work for now.

Masks are created in the same directory as the runs subdirectory. Structure for reference file and newly created masks/images by SAM:
```
├── some_folder_on_computer
|   |── some_robot_configuration    # data for a set of tubes
|       |- reference.csv            # reference file
|       |- input_prompts.json       # created by this program
|       |- run1                     # different folders per run as data collection could be done over several days
|           |── imgs                # images
|               |- filename1_cam0.jpg
|               |- filename1_cam1.jpg
|           |- masks                    # created by this program
|               |- mask_filename1_cam0.jpg
|               |- mask_filename1_cam1.jpg
|               |- ...
|           |- image_and_masks          # created by this program (not default)
|               |- ...
|       |── run2
|           |── imgs
|               |- filename10001_cam0.jpg
|               |- filename10001_cam1.jpg
|               |- ...
|           |── masks
|               |- mask_filename11001_cam0.jpg
|               |- mask_filename11001_cam1.jpg
|               |- ...
|           |- image_and_masks          # created by this program (not default)
|               |- ...
|       |── run3 ...
```

One single reference csv file should hold all the relative paths to each stereoimage pair with at least the following columns:

| frame_id      | batch_num | left_image_path | right_image_path |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | 0 | /imgs/run1/0/filename1_cam0.jpg | /imgs/run1/0/filename1_cam1.jpg |
| 2 | 0 | /imgs/run1/0/filename2_cam0.jpg | /imgs/run1/0/filename2_cam1.jpg |
| ... | ... | ... | ... |

# Run
To run program, with an input prompt generation app:
```bash
python main.py --config config/config.yaml --data-path /path/to/reference/file/folder --use-gui True 
```

or if you have an input prompt json in the data folder, to load:
```bash
python main.py --config config/config.yaml --data-path /path/to/reference/file/folder --use-gui True --input-prompt-json-name input_prompts.json
```

# Usage
Refer to [reference doc]

# Authors
Authors are listed in alphabetic order.

# License
BSD 3-Clause License

# BibTeX
If you want to reference this project, you can use the following citation:


# TODO:
- update prompts online with mask generation midway -- but background batching?
- should work without image batches in paths?
- pyhton recon? all embeded into gui
- usage doc
- clean up code

desired workflow:
- more integrated workflow? e.g., all in python with minimal work or run matlab in python
    - automated reconstruction: calibration of xy-plane with qr code on the trocar face
- flexibility: allow use of ui or offline pipelines
    - ours: offline processing then verify images (and note bad masks/reconstructions) then UI troubleshooting (fix masks or reconstructions by new input prompts)
