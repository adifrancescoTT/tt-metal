# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    pad_and_fold_conv_filters_for_unity_stride,
    pad_and_fold_conv_activation_for_unity_stride,
)

from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    preprocess_remaining_children_and_parameters,
    convert_torch_model_to_ttnn_model,
)


def ResnetLinear(
    in_features: int,
    out_features: int,
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
    transpose: bool = True,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
    model_config=None,
    device=None,
    batch_size=None,
):
    """
    Returns a function for linear operation in resnet with bias.
    """
    if bias is not None:
        assert bias.get_legacy_shape()[-1] == out_features, "bias shape is not as expected"
        if device is not None:
            bias = bias.to(device)

    if transpose:
        assert weight.get_legacy_shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"
        weight_T = tt_lib.tensor.transpose(weight, -2, -1)
    else:
        assert weight.get_legacy_shape() == [1, 1, in_features, out_features], "weight does not have the expected shape"
        weight_T = weight
    if device is not None:
        weight_T = weight_T.to(device)

    matmul_config = None
    if batch_size in hardcoded_matmul_config_linear and output_mem_config.is_sharded():
        matmul_config = hardcoded_matmul_config_linear[batch_size]

    def linear_(act):
        if is_grayskull():
            compute_kernel_config = tt_lib.tensor.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        ## this uses the systolic 1d matmul with bias fused
        if matmul_config is None:
            output = tt_lib.tensor.resnet_matmul(act, weight_T, bias, output_mem_config)
        else:
            output = tt_lib.operations.primary.matmul_1d(
                act,
                weight_T,
                bias=bias,
                program_config=matmul_config,
                output_mem_config=output_mem_config,
                output_dtype=model_config["ACTIVATIONS_DTYPE"],
                compute_kernel_config=compute_kernel_config,
            )
        return output

    return linear_


def do_nothing_op(x):
    return x


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.get_layout() == target_layout:
        return x
    if x.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.get_legacy_shape(), False, False, True, True)
        if x.get_legacy_shape() != x_padded_shape:
            return tt_lib.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.tilize(x, output_mem_config, use_multicore=True)
    elif x.get_layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        if x.get_legacy_shape() != x.shape_without_padding():
            return tt_lib.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.untilize(x, output_mem_config, use_multicore=True)
    else:
        assert False


# Local copy of unpad_from_zero to always set output to
def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != tt_lib.tensor.Layout.ROW_MAJOR:
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        x = x.to_torch().to(torch.float)
    return x


def compute_conv_output_shape(conv_params, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    OH = ((int)((H - R + 2 * P_H) / U)) + 1
    OW = ((int)((W - S + 2 * P_W) / V)) + 1
    return [x_shape[0], OH, OW, K]


# TODO: this function is required because conv is preprocessed before in TTNN model preprocessing flow
# We need to skip conv preprocessing there
def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


class resnet50Bottleneck:
    expansion: int = 4

    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        stride,
        sharded_memory_config_type,
        downsample=None,
        model_config=None,
        conv_2d=False,
        module_out_sharded=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.model_config = model_config
        self.output_memory_config = sharded_memory_config_type if module_out_sharded else ttnn.L1_MEMORY_CONFIG
        self.out_in_place = module_out_sharded
        parameters.conv1.weight, parameters.conv1.bias = permute_conv_weights(
            parameters.conv1.weight, parameters.conv1.bias
        )
        parameters.conv2.weight, parameters.conv2.bias = permute_conv_weights(
            parameters.conv2.weight, parameters.conv2.bias
        )
        parameters.conv3.weight, parameters.conv3.bias = permute_conv_weights(
            parameters.conv3.weight, parameters.conv3.bias
        )
        parameters.conv1.weight, parameters.conv1.bias = fold_batch_norm2d_into_conv2d(parameters.conv1, parameters.bn1)
        parameters.conv2.weight, parameters.conv2.bias = fold_batch_norm2d_into_conv2d(parameters.conv2, parameters.bn2)
        parameters.conv3.weight, parameters.conv3.bias = fold_batch_norm2d_into_conv2d(parameters.conv3, parameters.bn3)

        conv1_in_channels = parameters.conv1.weight.shape[1]

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.module_input_shape = [batch_size, input_height, input_width, conv1_in_channels]
        self.deallocate = True
        self.downsample_or_noop = downsample
        if self.downsample_or_noop is None:
            self.downsample_or_noop = do_nothing_op
            self.deallocate = False

        # 1x1 conv with stride 1 padding 0 is run using regular matmul
        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        out_channels = parameters.conv1.weight.shape[0]
        in_channels = parameters.conv1.weight.shape[1]
        self.conv1 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            conv_blocking_and_parallelization_config_override={},
            compute_kernel_config=compute_kernel_config,
            activation="relu",
        )

        move_utwh_output = False
        if self.deallocate and (
            self.module_input_shape[0] == 20 and self.module_input_shape[1] == 56 and self.module_input_shape[3] == 256
        ):
            move_utwh_output = True
        out_channels = parameters.conv2.weight.shape[0]
        in_channels = parameters.conv2.weight.shape[1]
        self.conv2 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv2.weight,
            bias=parameters.conv2.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            reallocate_halo_output=move_utwh_output,
            deallocate_activation=True,
            conv_blocking_and_parallelization_config_override={},
            compute_kernel_config=compute_kernel_config,
            activation="relu",
        )

        input_height = ((int)((input_height - 1) / stride)) + 1
        input_width = ((int)((input_width - 1) / stride)) + 1
        out_channels = parameters.conv3.weight.shape[0]
        in_channels = parameters.conv3.weight.shape[1]
        self.conv3 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv3.weight,
            bias=parameters.conv3.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            conv_blocking_and_parallelization_config_override={},
            compute_kernel_config=compute_kernel_config,
        )

    def resnet50_bottleneck_block(self, x):
        # logger.info("This module input shape - ", self.module_input_shape)
        # conv1 is 1x1 conv
        # logger.info("Running conv1")
        out = self.conv1(x)

        if not (self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64):
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                ttnn.deallocate(x)

        # logger.info("Running conv2")
        out = self.conv2(out)
        # conv3 is 1x1 conv
        # logger.info("Running conv3")
        out = self.conv3(out)

        if self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64:
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                ttnn.deallocate(x)

        out = ttnn.add_and_apply_activation(
            out, ds_out, activation="relu", memory_config=self.output_memory_config, out_in_place=self.out_in_place
        )

        if self.module_input_shape[0] == 20 and self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64:
            out = ttnn.experimental.tensor.move_sharded(out)

        return out


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        num_classes = 1000
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.reader_patterns_cache = {}
        self.inplanes = 64
        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        parameters.conv1.weight, parameters.conv1.bias = permute_conv_weights(
            parameters.conv1.weight, parameters.conv1.bias
        )
        parameters.conv1.weight, parameters.conv1.bias = fold_batch_norm2d_into_conv2d(parameters.conv1, parameters.bn1)
        parameters.conv1.weight = pad_and_fold_conv_filters_for_unity_stride(parameters.conv1.weight, 2, 2)
        out_channels = parameters.conv1.weight.shape[0]
        in_channels = parameters.conv1.weight.shape[1]
        self.conv1 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=True,
            batch_size=batch_size,
            input_height=115,
            input_width=115,
            reader_patterns_cache=self.reader_patterns_cache,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            use_shallow_conv_variant=True,
            deallocate_activation=True,
            padded_input_channels=16,
            activation="relu",
            conv_blocking_and_parallelization_config_override={},
            compute_kernel_config=compute_kernel_config,
        )

        self.max_pool_reader_patterns_cache = {}
        self.max_pool = ttnn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            pad=(1, 1),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=self.device,
            batch_size=self.batch_size,
            input_height=112,
            input_width=112,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
        )

        self.layer1, self.layer1_output_shape = self._make_layer(
            block,
            64,
            layers[0],
            name="layer1",
            state_dict=state_dict,
            layer_input_shape=self.maxpool_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=True,
            conv_halo=True if sharded else False,
            model_config=model_config,
        )
        self.layer2, self.layer2_output_shape = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            name="layer2",
            state_dict=state_dict,
            layer_input_shape=self.layer1_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=False,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            conv_halo=True if sharded else False,
            model_config=model_config,
        )
        self.layer3, self.layer3_output_shape = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            name="layer3",
            state_dict=state_dict,
            layer_input_shape=self.layer2_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=False,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            model_config=model_config,
            conv_halo=True if sharded else False,
            conv_2d=True,
        )
        self.layer4, self.layer4_output_shape = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            name="layer4",
            state_dict=state_dict,
            layer_input_shape=self.layer3_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=True,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            model_config=model_config,
            conv_halo=True if sharded else False,
            conv_2d=True,
        )

        # All modules in RN50 are unrolled here. One variable for each module. Only specific number of modules supported - layers MUST equal to [3, 4, 6, 3]
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.avgpool = TtAvgPool(self.device)

        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = torch.transpose(fc_weight, 3, 2)
        fc_weight = tt_lib.tensor.Tensor(
            fc_weight.reshape(-1).tolist(),
            fc_weight.shape,
            model_config["WEIGHTS_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(tt_lib.tensor.Layout.TILE)
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = tt_lib.tensor.Tensor(
            fc_bias.reshape(-1).tolist(), fc_bias.shape, model_config["WEIGHTS_DTYPE"], tt_lib.tensor.Layout.ROW_MAJOR
        ).to(tt_lib.tensor.Layout.TILE)
        self.fc = ResnetLinear(
            512 * block.expansion,
            1024,
            fc_weight,
            fc_bias,
            transpose=False,
            output_mem_config=self.width_sharded_memory_config,
            model_config=model_config,
            device=self.device,
            batch_size=batch_size,
        )  # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def __del__(self):
        # Need to clear global configs for each Resnet run
        self.reader_patterns_cache.clear()
        self.max_pool_reader_patterns_cache.clear()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        sharded_memory_config_type,
        module_out_sharded,
        model_config=None,
        conv_2d=False,
    ):
        downsample = None
        self.downsample_conv_on_tt = None
        self.norm_layer_after_downsample_conv_on_tt = None

        if stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion:
            nl = norm_layer(planes * block.expansion)
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"]
            )
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"],
                requires_grad=False,
            )
            nl.eval()
            downsample_conv_weight = state_dict[f"{self.base_address_with_dot}{name}.0.downsample.0.weight"]
            downsample_conv_bias = None

            if self.fold_batchnorm:
                downsample_conv_weight, downsample_conv_bias = fold_bn_to_conv_weights_bias(downsample_conv_weight, nl)
                nl = nn.Identity()

            # With single buffered input CB, these shapes work -
            # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv = {
            #     (3136, 256) : [128, 128, 128, 64] ,
            #     (800, 512) : [128, 128, 128, 64] ,
            #     (224, 1024) : [64, 128, 64, 64],
            #     (64, 2048) : [64, 128, 64, 64] ,
            # }

            downsample_output_channels = planes * resnet50Bottleneck.expansion
            self.downsample_params = [
                downsample_output_channels,
                self.inplanes,
                1,
                1,
                stride,
                stride,
                0,
                0,
                self.dilation,
                1,
            ]
            self.downsample_conv_output_shape = compute_conv_output_shape(self.downsample_params, layer_input_shape)
            is_downsample_1x1_conv = stride == 1
            is_1x1_downsample_conv_sanity_check = (
                self.downsample_params[2] == 1
                and self.downsample_params[3] == 1
                and self.downsample_params[4] == 1
                and self.downsample_params[5] == 1
                and self.downsample_params[6] == 0
                and self.downsample_params[7] == 0
            )
            assert is_1x1_downsample_conv_sanity_check == is_downsample_1x1_conv
            downsample_output_padded_face_size = _nearest_32(
                self.downsample_conv_output_shape[0]
                * self.downsample_conv_output_shape[1]
                * self.downsample_conv_output_shape[2]
            )
            matmul_config = None

            if is_grayskull():
                compute_kernel_config = tt_lib.tensor.GrayskullComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                )
            else:
                compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )

            if is_downsample_1x1_conv:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                # logger.info("Setting matmul config for 1x1 conv (downsample stride 1 conv in module)")
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                self.downsample_conv_on_tt = resnet50_1x1_conv_as_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    output_mem_config=self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    compute_kernel_config=compute_kernel_config,
                )
            elif use_downsample_op_and_mm_for_conv1x1_s2:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                assert stride == 2
                downsample_op_params = [batch_size, layer_input_shape[1], layer_input_shape[2], stride, stride]
                # logger.info("Calling ds op and matmul op, input shape - ", layer_input_shape)

                self.downsample_conv_on_tt = resnet50_1x1_conv_s2_as_downsample_and_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    downsample_op_params,  # used by downsample op
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                assert (
                    downsample_output_padded_face_size,
                    downsample_output_channels,
                ) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
                [
                    act_block_w_datums,
                    act_block_h_datums,
                    weight_block_w_datums,
                    out_subblock_h_datums,
                    out_subblock_w_datums,
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h,
                    per_core_weight_w,
                    num_cores_nhw,  # This number is only meaningful for batch 8, 16
                ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][
                    (downsample_output_padded_face_size, downsample_output_channels)
                ]
                assert per_core_act_h % 32 == 0
                per_core_act_h_ntiles = (int)(per_core_act_h / 32)
                per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
                assert self.inplanes % act_block_w_datums == 0
                self.downsample_conv_on_tt = resnet50_optimized_conv(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    [act_block_h_datums, act_block_w_datums],
                    [act_block_w_datums, weight_block_w_datums],
                    [out_subblock_h_datums, out_subblock_w_datums],
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h_ntiles,
                    per_core_weight_w_ntiles,
                    downsample_conv_bias.tolist(),
                    output_mem_config=self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    math_fidelity=model_config["MATH_FIDELITY"],
                    compute_kernel_config=compute_kernel_config,
                )
            self.norm_layer_after_downsample_conv_on_tt = nl

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                device=self.device,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0",
                fold_batchnorm=self.fold_batchnorm,
                downsample_conv_on_tt=self.downsample_conv_on_tt,
                norm_layer_after_downsample_conv_on_tt=self.norm_layer_after_downsample_conv_on_tt,
                downsample_params=self.downsample_params,
                storage_in_dram=self.storage_in_dram,
                input_shape=layer_input_shape,
                batch_size=batch_size,
                sharded=sharded,
                out_sharded=sharded is not None,
                use_downsample_op_and_mm_for_conv1x1_s2=use_downsample_op_and_mm_for_conv1x1_s2,
                model_config=model_config,
                conv_halo=conv_halo,
                conv_2d=conv_2d,
                reader_patterns_cache=self.reader_patterns_cache,
            )
        )
        self.inplanes = planes * block.expansion
        for block_num in range(1, blocks):
            previous_layer = layers[-1]
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    device=self.device,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{block_num}",
                    fold_batchnorm=self.fold_batchnorm,
                    storage_in_dram=self.storage_in_dram,
                    input_shape=previous_layer.conv3_output_shape,
                    batch_size=batch_size,
                    sharded=sharded,
                    out_sharded=True if block_num != blocks - 1 else out_sharded,
                    model_config=model_config,
                    conv_halo=conv_halo,
                    conv_2d=conv_2d,
                    reader_patterns_cache=self.reader_patterns_cache,
                )
            )
        last_layer_shape = layers[-1].conv3_output_shape
        return layers, last_layer_shape

    def preprocessing(self, x: torch.Tensor) -> tt_lib.tensor:
        if self.sharded:
            x = pad_and_fold_conv_activation_for_unity_stride(x, 3, 3, 2, 2)
            x = torch.permute(x, (0, 2, 3, 1))
            x = x.reshape(
                1,
                1,
                x.shape[0] * x.shape[1] * x.shape[2],
                x.shape[3],
            )
            input_size_to_shard_evenly = _nearest_y(x.shape[2], self.first_conv_num_cores_nhw * 32)
            x = torch.nn.functional.pad(x, (0, 0, 0, input_size_to_shard_evenly - x.shape[2], 0, 0))

            x = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16)
        else:
            extra_padding_for_32B_alignment = 25
            x = torch.nn.functional.pad(x, (3, 4 + extra_padding_for_32B_alignment, 3, 3, 0, 1))
            x = torch.permute(x, (0, 2, 3, 1))
            x = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16)
        return x

    def preprocessing_with_fold(self, x: torch.Tensor) -> tt_lib.tensor:
        if not self.sharded:
            raise ValueError("Preprocessing is only supported for sharded model")

        stride_h = 2
        stride_w = 2

        # NCWH -> NWHC
        x = torch.permute(x, (0, 2, 3, 1))

        # pad to 230x230x4
        C = _nearest_y(x.shape[3], 4)
        x = torch.nn.functional.pad(x, (0, C - x.shape[3], 3, 3, 3, 3))

        # reshape to [N, H, W / stride_w, C * stride_w]. It's a no-op in torch, and this way we only need to fold on height.
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // stride_w, x.shape[3] * stride_w)

        NHW = x.shape[0] * x.shape[1] * x.shape[2]
        NHW_even = _nearest_y(NHW // stride_h, self.first_conv_num_cores_nhw * 32)

        shard_spec = tt_lib.tensor.ShardSpec(
            self.fold_grid, [NHW // self.n_fold_cores, x.shape[3]], tt_lib.tensor.ShardOrientation.ROW_MAJOR, False
        )
        x = torch2tt_tensor(
            x,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                tt_lib.tensor.BufferType.L1,
                shard_spec,
            ),
        )

        # fold for unity stride on device
        x = tt_lib.tensor.fold(x, stride_h=stride_h, stride_w=1)

        shard_shape = [
            NHW_even // self.first_conv_num_cores_nhw,
            x.get_legacy_shape()[3],
        ]

        x = tt_lib.tensor.reshard(
            x,
            tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                tt_lib.tensor.BufferType.L1,
                tt_lib.tensor.ShardSpec(
                    self.shard_grid,
                    shard_shape,
                    tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        )

        return x

    def forward(self, x: tt_lib.tensor) -> tt_lib.tensor:
        if not self.sharded:
            original_A_cl_host_shape = x.get_legacy_shape()
            x = x.reshape(
                x.get_legacy_shape()[0], x.get_legacy_shape()[1], 1, x.get_legacy_shape()[2] * x.get_legacy_shape()[3]
            )

            x = x.to(self.device, self.memory_config)  # to l1
            # re-shape back to original shape (N, H, W, C)
            x = x.reshape(
                original_A_cl_host_shape[0],
                original_A_cl_host_shape[1],
                original_A_cl_host_shape[2],
                original_A_cl_host_shape[3],
            )
        elif x.storage_type() != tt_lib.tensor.StorageType.DEVICE:
            shard_spec = tt_lib.tensor.ShardSpec(
                self.shard_grid,
                [
                    x.get_legacy_shape()[2] // self.first_conv_num_cores_nhw,
                    x.get_legacy_shape()[3],
                ],
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, shard_spec
            )
            x = x.to(self.device, mem_config)

        x = self.conv1(x)
        # Relu is fused with conv1

        if self.batch_size == 20:
            x = tt_lib.tensor.move_sharded(x)

        if not self.sharded:
            x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
            x = x.reshape(
                self.conv1_output_shape[0],
                self.conv1_output_shape[1],
                self.conv1_output_shape[2],
                self.conv1_output_shape[3],
            )
        x = self.maxpool(x)

        x = x.reshape(
            1,
            1,
            self.maxpool_output_shape[0] * self.maxpool_output_shape[1] * self.maxpool_output_shape[2],
            self.maxpool_output_shape[3],
        )
        x = tt_lib.tensor.tilize(
            x,
            output_mem_config=self.height_sharded_memory_config,
            output_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            use_multicore=True,
        )
        if self.batch_size == 20:
            x = tt_lib.tensor.move_sharded(x)

        x = self.layer1_module1(x)
        x = self.layer1_module2(x)
        x = self.layer1_module3(x)

        x = self.layer2_module1(x)
        x = self.layer2_module2(x)
        x = self.layer2_module3(x)
        x = self.layer2_module4(x)
        if self.sharded:
            grid_size = (10, 8)
            x = tt_lib.tensor.interleaved_to_sharded(
                x,
                self.layer_3_grid_size,
                [
                    math.ceil((x.get_legacy_shape()[-2] // 32) / self.layer_3_grid_size[0]) * 32,
                    x.get_legacy_shape()[-1] // self.layer_3_grid_size[1],
                ],
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
            )
        x = self.layer3_module1(x)
        x = self.layer3_module2(x)
        x = self.layer3_module3(x)
        x = self.layer3_module4(x)
        x = self.layer3_module5(x)
        x = self.layer3_module6(x)
        if self.sharded:
            x = tt_lib.tensor.interleaved_to_sharded(
                x,
                self.layer_4_grid_size,
                [
                    math.ceil((x.get_legacy_shape()[-2] // 32) / self.layer_4_grid_size[0]) * 32,
                    x.get_legacy_shape()[-1] // self.layer_4_grid_size[1],
                ],
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
            )
        x = self.layer4_module1(x)
        x = self.layer4_module2(x)
        x = self.layer4_module3(x)

        unpadded_shape = x.shape_without_padding()
        x = tt_lib.tensor.untilize_with_unpadding(
            x,
            (0, 0, 0, 0),
            (unpadded_shape[0] - 1, unpadded_shape[1] - 1, unpadded_shape[2] - 1, unpadded_shape[3] - 1),
            self.memory_config,
        )

        x = x.reshape(
            self.batch_size,
            x.get_legacy_shape()[1],
            (int)(x.get_legacy_shape()[2] / self.batch_size),
            x.get_legacy_shape()[3],
        )
        if self.sharded:
            grid_size = (8, 4)
            x = tt_lib.tensor.interleaved_to_sharded(
                x,
                grid_size,
                [x.volume() // x.get_legacy_shape()[-1], x.get_legacy_shape()[-1] // (grid_size[0] * grid_size[1])],
                tt_lib.tensor.TensorMemoryLayout.WIDTH_SHARDED,
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            )

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        if self.sharded:
            x = tt_lib.tensor.tilize_with_val_padding(
                x,
                padded_shape,
                [0, 0, 0, 0],
                0,
                output_mem_config=self.width_sharded_memory_config,
                output_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        else:
            x = tt_lib.tensor.pad(
                x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config, use_multicore=True
            )
            x = tt_lib.tensor.tilize(
                x,
                output_mem_config=self.memory_config,
                output_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                use_multicore=True,
            )

        x = self.avgpool(x, self.width_sharded_memory_config)

        unpadded_shape_end = [
            x.get_legacy_shape()[0] - 1,
            x.get_legacy_shape()[1] - 1,
            1 - 1,
            x.get_legacy_shape()[3] - 1,
        ]
        if self.sharded:
            x = tt_lib.tensor.untilize_with_unpadding(
                x, (0, 0, 0, 0), unpadded_shape_end, output_mem_config=self.width_sharded_memory_config
            )
        else:
            x = tt_lib.tensor.untilize(x, self.memory_config, use_multicore=True)
            x = tt_lib.tensor.unpad(x, (0, 0, 0, 0), unpadded_shape_end, output_mem_config=self.memory_config)

        x = x.reshape(1, x.get_legacy_shape()[1], self.batch_size * x.get_legacy_shape()[2], x.get_legacy_shape()[3])

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        if self.sharded:
            x = tt_lib.tensor.tilize_with_val_padding(
                x,
                padded_shape,
                [0, 0, 0, 0],
                0,
                output_mem_config=self.width_sharded_memory_config,
                output_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        else:
            x = tt_lib.tensor.pad(
                x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config, use_multicore=True
            )
            x = tt_lib.tensor.tilize(
                x,
                output_mem_config=self.memory_config,
                output_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                use_multicore=True,
            )

        x = self.fc(x)
        desired_shape = list(x.shape_without_padding())
        desired_shape[-1] = 1000
        x = tt_lib.tensor.untilize_with_unpadding(
            x,
            [0, 0, 0, 0],
            (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1),
            self.memory_config,
        )
        x = x.reshape(
            self.batch_size,
            x.get_legacy_shape()[1],
            (int)(x.get_legacy_shape()[2] / self.batch_size),
            x.get_legacy_shape()[3],
        )

        return x
