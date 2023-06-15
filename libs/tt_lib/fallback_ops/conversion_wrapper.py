from .. import tensor as ttl_tensor, device as ttl_device
import torch
from functools import wraps
from loguru import logger
import os

_pytorch_warning_logged = False

# Log only once to not pollute output
def check_log_pytorch_warning(arg):
    global _pytorch_warning_logged
    if not _pytorch_warning_logged and torch.is_tensor(arg):
        logger.warning(
            "Pytorch tensor was passed as input to fallback op instead of TT tensor. This is currently supported to improve perf but support for this will be deprecated."
        )
        _pytorch_warning_logged = True

def convert_tt_tensor_to_pt_tensor(tt_tensor, host, output_format):
    # Update output_format with format of first encountered arg
    if output_format.get("device", None) is None and not tt_tensor.on_host():
        output_format["device"] = tt_tensor.device()

    if output_format.get("dtype", None) is None:
        output_format["dtype"] = tt_tensor.dtype()

    # Convert to PT Tensor
    if not tt_tensor.on_host():
        tt_tensor = tt_tensor.to(host)

    if tt_tensor.layout() != ttl_tensor.Layout.ROW_MAJOR:
        tt_tensor = tt_tensor.to(ttl_tensor.Layout.ROW_MAJOR)

    return torch.Tensor(tt_tensor.data()).reshape(tt_tensor.shape())


def convert_pt_tensor_to_tt_tensor(pt_tensor, output_format):
    tt_tensor = ttl_tensor.Tensor(
        pt_tensor.reshape(-1).tolist(),
        pt_tensor.shape,
        output_format["dtype"],
        ttl_tensor.Layout.ROW_MAJOR,
    )

    if output_format["layout"] == ttl_tensor.Layout.TILE:
        if (
            tt_tensor.shape()[2] % 32 == 0 and tt_tensor.shape()[3] % 32 == 0
        ):  # Restore tile layout only if legal or else leave as RM
            tt_tensor = tt_tensor.to(ttl_tensor.Layout.TILE)
    else:
        if output_format["layout"] != ttl_tensor.Layout.ROW_MAJOR:
            tt_tensor = tt_tensor.to(output_format["layout"])

    if isinstance(output_format["device"], ttl_device.Device):
        if (
            tt_tensor.layout() == ttl_tensor.Layout.TILE
            or tt_tensor.layout() == ttl_tensor.Layout.ROW_MAJOR
            and tt_tensor.shape()[3] % 2 == 0
            or tt_tensor.layout() == ttl_tensor.Layout.CHANNELS_LAST
            and tt_tensor.shape()[1] % 2 == 0
        ):
            tt_tensor = tt_tensor.to(output_format["device"])
    return tt_tensor


def convert_tt_tensors_to_pt_tensors(args, host, output_format):
    check_log_pytorch_warning(args)
    if isinstance(args, ttl_tensor.Tensor):
        return convert_tt_tensor_to_pt_tensor(args, host, output_format)
    elif isinstance(args, dict):
        outputs = {}
        for key, value in args.items():
            if isinstance(value, ttl_tensor.Tensor):
                outputs[key] = convert_tt_tensor_to_pt_tensor(
                    value, host, output_format
                )
            elif isinstance(value, (list, tuple, dict)):
                outputs[key] = convert_tt_tensors_to_pt_tensors(
                    value, host, output_format
                )
            else:
                check_log_pytorch_warning(args)
                outputs[key] = value
        return outputs
    elif isinstance(args, (list, tuple, dict)):
        outputs = []
        for arg in args:
            if isinstance(arg, ttl_tensor.Tensor):
                outputs.append(convert_tt_tensor_to_pt_tensor(arg, host, output_format))
            elif isinstance(arg, (list, tuple, dict)):
                outputs.append(
                    convert_tt_tensors_to_pt_tensors(arg, host, output_format)
                )
            else:
                check_log_pytorch_warning(args)
                outputs.append(arg)
        return outputs
    else:
        return args


def convert_pt_tensors_to_tt_tensors(args, output_format):
    if isinstance(args, torch.Tensor):
        return convert_pt_tensor_to_tt_tensor(args, output_format)
    elif isinstance(args, dict):
        outputs = []
        for key, value in args.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = convert_pt_tensor_to_tt_tensor(value, output_format)
            elif isinstance(value, (list, tuple, dict)):
                outputs[key] = convert_pt_tensors_to_tt_tensors(value, output_format)
            else:
                outputs[key] = value
        return outputs
    elif isinstance(args, (list, tuple, dict)):
        outputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                outputs.append(convert_pt_tensor_to_tt_tensor(arg, output_format))
            elif isinstance(arg, (list, tuple, dict)):
                outputs.append(convert_pt_tensors_to_tt_tensors(arg, output_format))
            else:
                outputs.append(arg)
        return outputs
    else:
        return args


def convert_tt_tensors_wrapper(func):
    host = ttl_device.GetHost()

    @wraps(func)
    def wrap(*args, **kwargs):
        output_format = {"layout": ttl_tensor.Layout.TILE}

        new_args = convert_tt_tensors_to_pt_tensors(args, host, output_format)

        new_kwargs = convert_tt_tensors_to_pt_tensors(kwargs, host, output_format)

        # Set default output format
        if output_format.get("device", None) is None:
            output_format["device"] = ttl_device.GetDefaultDevice()
        if output_format.get("dtype", None) is None:
            output_format["dtype"] = ttl_tensor.DataType.BFLOAT16

        outputs = func(*new_args, **new_kwargs)

        # Convert pt tensors in outputs to tt tensors
        return convert_pt_tensors_to_tt_tensors(outputs, output_format)

    return wrap
