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

import sys
import os

# Add the `datasets` directory to the import path
current_dir = os.path.dirname(os.path.abspath(__file__))

print('current_dir' , current_dir)

datasets_dir = os.path.join(current_dir, 'datasets')
sys.path.append(current_dir)


import fsns
import fsns_test
import number_plates

__all__ = [fsns, fsns_test, number_plates]
