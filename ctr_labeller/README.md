
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

This program requires you to give the path to the folder with a reference.csv. Your reference csv can allow different image saving folder structures
but requires that the images are saved under an "imgs" folder because the program will use the specific string to create a "masks" folder beside it. 
It will then save masks with the same subfolder structure as the images.
We suggest the following structure for the reference file and saving of images and masks:
```
├── some_folder_on_computer
|   |── some_robot_configuration    # data for a set of tubes
|       |- reference.csv            # reference file
|       |- input_prompts.json       # created by this program
|       |- run1                     # different folders per run as data collection could be done over several days
|           |── imgs                # images
|               |-0 
|               |    |- filename1_cam0.jpg
|               |    |- filename1_cam1.jpg
|               |    |- ...
|               |-1
|                    |- ...
|           |- masks                    # created by this program
|               |-0
|               |    |- mask_filename1_cam0.jpg
|               |    |- mask_filename1_cam1.jpg
|               |    |- ...
|               |-1
|                    |- ...
|           |- image_and_masks          # created by this program (not default)
|               |- ...
|       |── run2
|           |── imgs                # images
|               |-0 
|               |    |- filename10001_cam0.jpg
|               |    |- filename10001_cam1.jpg
|               |    |- ...
|               |-1
|                    |- ...
|           |- masks                    # created by this program
|               |-0
|               |    |- mask_filename10001_cam0.jpg
|               |    |- mask_filename10001_cam1.jpg
|               |    |- ...
|               |-1
|                    |- ...
|           |- image_and_masks          # created by this program (not default)
|               |- ...
|       |── run3 ...
```

One single reference csv file should hold all the relative paths to each stereoimage pair with at least the following columns:

| frame_id | left_image_path | right_image_path |
| ----------- | ----------- | ----------- |
| 1 | /run1/imgs/0/filename1_cam0.jpg | /run1/imgs/0/filename1_cam1.jpg |
| 2 | /run1/imgs/0/filename2_cam0.jpg | /run1/imgs/0/filename2_cam1.jpg |
| ... | ... | ... |

# Run
To run program, with an input prompt generation app:
```bash
python main.py --config config/config.yaml --data-path /path/to/reference/file/folder --use-gui True 
```

or if you have an input prompt json in the data folder, to load:
```bash
python main.py --config config/config.yaml --data-path /path/to/reference/file/folder --use-gui True --input-prompt-json-name input_prompts.json
```
# License
BSD 3-Clause License

# BibTeX
If you want to reference this project, you can use the following citation:
```bibtex
    @INPROCEEDINGS{kang_ismr_2025,
      author={Kang, Paul H. and Gondokaryono, Radian and Roshanfar, Majid and Nguyen, Robert H. and Looi, Thomas and Drake, James M. and Podolsky, Dale},
      booktitle={2025 International Symposium on Medical Robotics (ISMR)}, 
      title={Learning Inverse Kinematics Multiplicity of Concentric Tube Robots Using Invertible Neural Networks}, 
      year={2025},
      volume={},
      number={},
      pages={1-7},
      keywords={},
      doi={}
    }
```
