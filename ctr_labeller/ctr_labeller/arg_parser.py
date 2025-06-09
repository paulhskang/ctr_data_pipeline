import argparse

def arg_parser():
    ''' load CLI arguments '''
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("--data-path",                      help="Directory containing reference file", type=str)
    # optional arguments
    parser.add_argument("--use-gui",                        help="User actively specifies which masks to save with the GUI", type=bool, default=True)
    parser.add_argument("--input-prompt-json-name",         help="Name of input prompt file", type=str, default="")
    parser.add_argument("--input-prompt-app-image-height",  help="Input prompt app image height", type=int, default=1080)
    parser.add_argument("--batch-num",                      help="Process this batch number only", type=int, default=-1)
    parser.add_argument("--save-image-appended-with-masks", help="Save images with mask overlays", type=bool, default=False)
    parser.add_argument("--sort-based-on",                  help="Criteria on how masks are selected", type=str, default="None")
    parser.add_argument("--max-size-to-add",                help="Number of images to load at one time for processing", type=int, default=40)

    args = parser.parse_args()
    return args