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
import sys
from pprint import pprint
from setuptools import setup, find_packages, find_namespace_packages
# import torch

# torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
# assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

excludes = []
packages = []
for p in find_namespace_packages(include='lsl_tools*'):
    if p.startswith('lsl_tools') and not p.startswith('lsl_tools.packages'):
        for excl in excludes:
            if excl in p:
                break
        else:
            packages.append(p)

# pprint(packages)
# sys.exit(0)

setup(
    name='lsl_tools',
    version='0.9.3',
    description='Internal software',
    url='',
    author='LSL team',
    license='MIT',
    packages=packages,
    install_requires=[
        "tabulate",
        "jinja2",
        "label-studio-converter",
        'pywin32 >= 1.0; platform_system=="Windows"'
    ],
    include_package_data=True,
    # scripts=["lsl_tools/lsl"],
    
    # To enable entry points and automatic script creation (lsl.exe in Windows)
    entry_points = {
        'console_scripts': [
            'lsl = lsl_tools.__main__:main',
        ],
    }
)
