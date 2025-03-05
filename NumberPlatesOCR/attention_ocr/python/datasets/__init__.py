# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

current_directory = os.getcwd()
print("Current working directory:", current_directory)

import sys
sys.path.append('/workspaces/codespaces-jupyter/NumberPlatesOCR/attention_ocr/python/datasets/')

import fsns
import fsns_test
import number_plates

__all__ = [fsns, fsns_test, number_plates]
