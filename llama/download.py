import os


here = os.path.dirname(os.path.realpath(__file__))


def download(args=None):
    import hiq

    cmd = f"bash {here}/download_community.sh"
    if args is not None:
        if args.model_size:
            cmd += f" {args.model_size}"
        if args.folder:
            cmd += f" {args.folder}"
    hiq.execute_cmd(cmd, verbose=False, shell=True, runtime_output=True)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        default="7B,13B,30B,65B",
        help='The size of the models that you want to download. A comma separated string of any of "7B", "13B", "30B", "65B". Totally 219G disk space is needed to download them all. If you only want to download the 7B model, just put "7B" here.',
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="pyllama_data",
        help="The target folder for the download files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    download(get_args())
