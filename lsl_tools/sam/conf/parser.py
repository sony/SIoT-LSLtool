import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument(
        "-c", "--config_file",
        default="configs/segm/sam_adapter.yaml", type=str,
        help="configs file",
    )
    parser.add_argument(
        "--only_test", action="store_true", default=False, help="whether train or test")
    parser.add_argument(
        "--gpu",  default=False, type=bool, help="whether use GPU")
    parser.add_argument(
        "--gpu_device", default=0, type=int, help="specify your GPU ids")
    parser.add_argument(
        '--distributed', action='store_true', default=False, help="use distributed training")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()
