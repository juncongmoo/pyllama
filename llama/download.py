import os
from threading import Thread


here = os.path.dirname(os.path.realpath(__file__))


def download(args=None):
    import hiq

    cmd = f"bash {here}/download_community.sh"
    if args is not None:
        if args.model_size:
            cmd += f" {args.model_size}"
        if args.folder:
            cmd += f" {args.folder}"
    retcode = hiq.execute_cmd(cmd, verbose=False, shell=True, runtime_output=True)
    if retcode != 0:
        # retry
        download(args)


def download_watchdog(args):
    def watch():
        import time

        # every 30s, check total file size under folder to see if it increases as the download speed suggests. if not, restart download
        folder = args.folder if args.folder else "pyllama_data"
        last_total_size = -1
        while True:
            total_size = 0
            for dirpath, _, filenames in os.walk(folder):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            size_changed_mb = (total_size - last_total_size) / 1024 / 1024
            if last_total_size != -1 and size_changed_mb < 30 * args.download_speed_mb:
                print(
                    f"Download watchdog: total file size {total_size / 1024 / 1024:.2f}MB increased too slow ({size_changed_mb:.2f}MB in the last 30s), restarting download"
                )
                import hiq

                cmd = f"bash {here}/download_community_stop.sh"
                hiq.execute_cmd(cmd, verbose=False, shell=True, runtime_output=True)
            else:
                if last_total_size != -1:
                    print(
                        f"Download watchdog: total file size increased normally at speed {size_changed_mb / 30:.2f}MB/s"
                    )
                last_total_size = total_size
            time.sleep(30)

    watch_thread = Thread(target=watch, daemon=True)
    watch_thread.start()


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
    parser.add_argument(
        "--download_speed_mb",
        type=int,
        default=1,
        help="The accepted download speed in MB/s. If the download speed is lower than this, the download will be restarted.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    download_watchdog(args)
    download(args)
