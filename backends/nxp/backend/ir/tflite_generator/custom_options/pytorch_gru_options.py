# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

import numpy as np
from flatbuffers import flexbuffers

from executorch.backends.nxp.backend.ir.tflite_generator.meta.meta import CustomOptions


class Activation(Enum):
    sigmoid = 'Sigmoid'
    tanh = 'Tanh'


class Direction(Enum):
    forward = 'forward'
    bidirectional = 'bidirectional'
    reverse = 'reverse'


class PyTorchGRU(CustomOptions):
    def __init__(
        self,
        hidden_size: int,
        clip: float = np.finfo(np.float32).max.item(),
        layout: int = 0,

        activations: tuple[Activation, Activation] = (Activation.sigmoid, Activation.tanh),
        # For `bidirectional`, the default is (Activation.sigmoid, Activation.tanh, Activation.sigmoid, Activation.tanh)
        # neutron-converter/src/GraphIR/GraphOptimizer.cpp#5345

        direction: Direction = Direction.forward,
        linear_before_reset: bool = True,
    ) -> None:
        # Replace the enums with their string values.
        activations_str = [act.value for act in activations]
        direction_str = direction.value

        custom_options_data = flexbuffers.Dumps({
            'activations': activations_str,
            'clip': clip,
            'direction': direction_str,
            'hidden_size': hidden_size,
            'layout': layout,
            'linear_before_reset': int(linear_before_reset)
        })

        super().__init__("PyTorchGRU", custom_options_data)
