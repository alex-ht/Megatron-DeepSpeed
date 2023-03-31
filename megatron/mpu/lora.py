import math
import torch
from torch import nn
import torch.nn.init as init
from typing import List, Optional, Union
import torch.nn.functional as F
from megatron import mpu
#from  import ColumnParallelLinear, RowParallelLinear
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class ColumnParallelLinear(mpu.ColumnParallelLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        input_size, output_size, bias=True, gather_output=True,
        init_method=init.xavier_normal_, stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        # fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        mpu.ColumnParallelLinear.__init__(self, input_size, output_size, bias=bias, gather_output=gather_output, init_method=init_method,
                     stride=stride, keep_master_weight_for_test=keep_master_weight_for_test, skip_bias_add=skip_bias_add, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(
                self.output_size_per_partition, r, bias=False)
            self.lora_B = nn.Linear(r, self.input_size, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.master_weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        mpu.ColumnParallelLinear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        mpu.ColumnParallelLinear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling

            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling
            self.merged = False

    def eval(self):
        mpu.ColumnParallelLinear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, input_: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling
                self.merged = False

            return CPL.forward(input_)
        elif self.r > 0 and not self.merged:
            output, output_bias = CPL.forward(input_)
            if self.r > 0:
                # Set up backprop all-reduce.
                input_parallel = copy_to_tensor_model_parallel_region(input_)
                # Matrix multiply.
                input_parallel = self.lora_dropout(input_parallel)
                tmp = F.linear(input_parallel, self.lora_B.weight, None)
                tmp = F.linear(tmp, self.lora_A.weight, None)
                output_parallel = tmp * self.scaling
                if self.gather_output:
                    # All-gather across the partitions.
                    output += gather_from_tensor_model_parallel_region(output_parallel)
                else:
                    output += output_parallel
            return output, output_bias
        else:
            return mpu.ColumnParallelLinear.forward(input_)


class RowParallelLinear(mpu.RowParallelLinear, LoraLayer):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 # fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 **kwargs,):
        mpu.ColumnParallelLinearRowParallelLinear.__init__(self, input_size, output_size, bias=bias, input_is_parallel=input_is_parallel, init_method=init_method,
                     stride=stride, keep_master_weight_for_test=keep_master_weight_for_test, skip_bias_add=skip_bias_add, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(
                self.output_size, r, bias=False)
            self.lora_B = nn.Linear(r, self.input_size_per_partition, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.master_weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        mpu.RowParallelLinear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        mpu.RowParallelLinear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling

            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling
            self.merged = False

    def eval(self):
        mpu.RowParallelLinear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, input_: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (self.lora_B.weight @
                                     self.lora_A.weight) * self.scaling
                self.merged = False

            return mpu.RowParallelLinear.forward(input_)
        elif self.r > 0 and not self.merged:
            output, output_bias = RowParallelLinear.forward(input_)
            if self.r > 0:
                # Set up backprop all-reduce.
                if self.input_is_parallel:
                    input_parallel = input_
                else:
                    input_parallel = scatter_to_tensor_model_parallel_region(input_)
                # Matrix multiply.
                input_parallel = self.lora_dropout(input_parallel)
                inner = F.linear(input_parallel, self.lora_B.weight)
                output_parallel = F.linear(inner, self.lora_A.weight)
                
                # All-reduce across all the partitions.
                output += reduce_from_tensor_model_parallel_region(output_parallel * self.scaling)
            return output, output_bias
        else:
            return mpu.RowParallelLinear.forward(input_)