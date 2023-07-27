import os
import csv
import sys
import time
import torch
import argparse
import yaml
import random
from pathlib import Path
from loguru import logger
from functools import partial
import tt_lib

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from python_api_testing.sweep_tests import comparison_funcs, generation_funcs

from python_api_testing.sweep_tests.common import (
    get_test_fieldnames,
    run_test_and_save_results,
    shapes_and_datagen,
)

from python_api_testing.sweep_tests.op_map import op_map

DTYPES_TT_DICT = {
    "BFLOAT16": tt_lib.tensor.DataType.BFLOAT16,
    "BFLOAT8_B": tt_lib.tensor.DataType.BFLOAT8_B,
}

LAYOUTS_TT_DICT = {
    "ROW_MAJOR": tt_lib.tensor.Layout.ROW_MAJOR,
    "TILE": tt_lib.tensor.Layout.TILE,
}


def run_pytorch_test(args):
    # Create output folder
    output_folder = Path(args.output_folder_path)

    # if output_folder.exists():
    #     logger.error(
    #         f"Directory {output_folder} already exists! Remove this folder or provide a different path to start pytorch tests."
    #     )
    #     sys.exit(1)

    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        write_to_csv = True
        logger.info(f"Starting pytorch tests in: {output_folder}. Writing to csv.")
    else:
        write_to_csv = False
        logger.info(f"Not logging results in {output_folder}. Delete that folder to write csv results.")

    ################# PARSE ARGS #################
    pcie_slot = args.pcie_slot
    logger.info(f"Running on device {pcie_slot} for test.")

    ################# PARSE TEST CONFIGS #################
    with open(args.input_test_config, "r") as stream:
        try:
            pytorch_test_configs_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)

    assert "test-list" in pytorch_test_configs_yaml
    pytorch_test_list = pytorch_test_configs_yaml["test-list"]

    default_env_dict = {
        "TT_PCI_DMA_BUF_SIZE": "1048576",
        "TT_METAL_DEVICE_DISPATCH_MODE": "1"
    }
    # Get env variables from CLI
    args_env_dict = {}
    if args.env != "":
        envs = args.env.split(" ")
        for e in envs:
            if "=" not in e:
                name = e
                value = "1"
            else:
                name, value = e.split("=")
            args_env_dict[name] = value

    # make list
    if isinstance(pytorch_test_list, dict):
        pytorch_test_list = [pytorch_test_list]

    start_time = time.time()
    run_id = 0
    random.seed(0)

    for i in range(len(pytorch_test_list)):
        for test_name, test_config in pytorch_test_list[i].items():

            assert test_name in op_map

            # Get env variables from yaml (yaml overrides CLI)
            yaml_env_dict = test_config.get("env", {})

            # Env variables to use (precedence yaml > cli > default)
            if yaml_env_dict:
                env_dict = yaml_env_dict
            elif args_env_dict:
                env_dict = args_env_dict
            else:
                env_dict = default_env_dict

            old_env_dict = {}
            assert isinstance(env_dict, dict)
            for key, value in env_dict.items():
                old_env_dict[key] = os.environ.pop(key, None)
                os.environ[key] = value

            shape_dict = test_config["shape"]
            datagen_dict = test_config["datagen"]
            results_csv_path = output_folder / test_config["output-file"]

            comparison_dict = test_config["comparison"]
            comparison_args = comparison_dict.get("args", {})
            comparison_func = partial(
                getattr(comparison_funcs, comparison_dict["function"]), **comparison_args
            )
            test_args_gen = getattr(
                generation_funcs,
                test_config.get("args-gen", "gen_default_dtype_layout_device"),
            )
            # Optional test args for dtype, etc...
            test_args = test_config.get("args", {})

            # Set tests parameters --------------------------
            # test layouts
            test_tt_layouts = []
            if 'data-layout' in test_args:
                for layout in test_args['data-layout']:
                    test_tt_layouts.append(LAYOUTS_TT_DICT[layout])
            else:
                test_tt_layouts = generation_funcs.supported_tt_layouts

            # test dtypes
            test_tt_dtypes = []
            if 'data-type' in test_args:
                for dtype in test_args['data-type']:
                    test_tt_dtypes.append(DTYPES_TT_DICT[dtype])
            else:
                test_tt_dtypes = generation_funcs.supported_tt_dtypes

            # is on device
            test_on_device_options = []
            if 'on-device' in test_args:
                for device_type in test_args['on-device']:
                    test_on_device_options.append(device_type)
            else:
                test_on_device_options = generation_funcs.on_device_options
            # Set tests parameters --------------------------

            skip_header = False
            if results_csv_path.exists():
                skip_header = True

            ################# RUN TEST SWEEP #################
            with open(results_csv_path, "a", newline="") as results_csv:
                results_csv_writer = None

                init_file = True
                for input_shapes, datagen_funcs in shapes_and_datagen(
                    shape_dict, datagen_dict
                ):
                    for generated_test_args in test_args_gen(input_shapes, test_tt_dtypes, test_tt_layouts, test_on_device_options):
                        generated_test_args.update(
                            test_args
                        )  # specified test args overrides generated test args

                        # Moved this here so that we don't need to maintain a hardcoded list of headers per op
                        if init_file and write_to_csv:
                            results_csv_writer = csv.DictWriter(
                                results_csv,
                                fieldnames=get_test_fieldnames(generated_test_args.keys()),
                            )
                            if not skip_header:
                                results_csv_writer.writeheader()
                                results_csv.flush()
                            init_file = False

                        data_seed = random.randint(0, 20000000) # int(time.time())
                        torch.manual_seed(data_seed)

                        logger.info(
                            f"Running with shape: {input_shapes} and seed: {data_seed}"
                        )

                        test_profiling_key = f"test_sweep_separator - {run_id}"
                        logger.info(f"Starting profiling test {test_profiling_key}")
                        tt_lib.profiler.start_profiling(test_profiling_key)

                        test_pass = run_test_and_save_results(
                            results_csv_writer,
                            test_name,
                            input_shapes,
                            data_seed,
                            env_dict,
                            generated_test_args,
                            op_map[test_name]["tt_lib_op"],
                            op_map[test_name]["pytorch_op"],
                            input_shapes,
                            datagen_funcs,
                            comparison_func,
                            pcie_slot,
                            generated_test_args,
                        )

                        tt_lib.device.Synchronize()
                        tt_lib.profiler.stop_profiling(test_profiling_key)
                        logger.info(f"Stopped profiling test {test_profiling_key}")
                        run_id += 1

                        results_csv.flush()

                        # Check if test passed
                        if args.run_tests_for_ci and not test_pass:
                            logger.error(
                                f"{test_name} test failed with input shape {input_shapes}."
                            )
                            sys.exit(1)

            # Unset env variables
            for key, value in old_env_dict.items():
                os.environ.pop(key)
                if value is not None:
                    os.environ[key] = value

    duration = time.time() - start_time
    logger.info(f"Tests run in {duration:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch testing infra")
    parser.add_argument(
        "-i",
        "--input-test-config",
        help="Input pytorch test config",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-folder-path",
        default="pytorch_test_folder",
        help="Output pytorch test folder",
    )
    parser.add_argument(
        "-s",
        "--pcie-slot",
        default=0,
        type=int,
        help="Virtual PCIE slot of GS device to run on",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="",
        help="Env variables to set",
    )
    parser.add_argument(
        "--run-tests-for-ci",
        action="store_true",
        help="If set, assert on test result after every test.",
    )
    args = parser.parse_args()

    run_pytorch_test(args)
