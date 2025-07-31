# Copyright 2025 Sony Group Corporation
#
# Redistribution and use in source and binary forms, with or without modification, are permitted 
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the 
# distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import base64
import io
from PIL import Image

import zerorpc

from lsl_tools.cvat.lsl_inference.sam import SamSegmentationFunction

class SamSegemtationSeverless(SamSegmentationFunction):
    def cvat_severless_results(self, model_results: dict):
        results = []
        for label, score, segm in zip(model_results["labels"], model_results["scores"], model_results["segmentations"]):
            results.append({
                    "confidence": str(float(score)),
                    "label": self.categories[label-1]["name"],
                    "mask":segm,
                    "type": "mask",
                    })
        return results

    def severless_predict(self, image, threshold: float):
        buf = io.BytesIO(base64.b64decode(image))
        image = Image.open(buf).convert('RGB')
        self.confidence_score = threshold
        model_results = self.predict(image)
        results = self.cvat_severless_results(model_results)
        return results

def make_parser():
    parser = argparse.ArgumentParser(description='Label studio')
    parser.add_argument(
        '-p', '--port', type=int, default=4042,
        help='Server port')
    parser.add_argument(
        '--config-file', dest='config_file', default="",
        help='the config path of SAM')
    parser.add_argument(
        "--conf", default=0.3, type=float,
        help="confidence score")
    parser.add_argument(
        "--mask-threshold", default=-1, type=float,
        help="mask threshold")
    parser.add_argument(
        '--project-dir', dest='project_dir', default="",
        help='the path of the lsl project')
    parser.add_argument(
        "--slidewindow", default=False, action="store_true",
        help="Slide windows control")
    parser.add_argument(
        "--slidewindow-size", default=(1024, 1024), type=int, nargs="+",
        help="(width, height)the cropped image size after slide windows")
    parser.add_argument(
        "--overlap", default=150, type=int,
        help="the overlap of the sliding windows and should be bigger than the height and width of instances")
    parser.add_argument(
        "--sam-only", default=False, type=bool,
        help="whether to use glip or not as box prompts generator")
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    server = zerorpc.Server(SamSegemtationSeverless(args, args.config_file, args.project_dir, args.conf, args.slidewindow, args.slidewindow_size, args.overlap, args.sam_only, cvat_mask=True))
    server.bind(f"tcp://0.0.0.0:{args.port}")
    server.run()