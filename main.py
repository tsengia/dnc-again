#!/usr/bin/env python3
#
# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Tyler Sengia.
#
# ==============================================================================



from lightning.pytorch.cli import LightningCLI

from models.dnc import LitDNC


class LengthHackSampler:
    def __init__(self, batch_size, length):
        self.length = length
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            len = self.length() if callable(self.length) else self.length
            yield [len] * self.batch_size

    def __len__(self):
        return 0x7FFFFFFF

def cli_main():
    cli = LightningCLI(LitDNC)

if __name__ == "__main__":

    cli_main()
