import argparse

# from utils.cloud_utils import upload_dir
from cloud_utils import upload_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str)
    parser.add_argument("--cloud_dir", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    upload_dir(args.local_dir, args.cloud_dir)


if __name__ == "__main__":
    main()
