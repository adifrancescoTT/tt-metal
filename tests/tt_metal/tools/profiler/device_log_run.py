import os, sys
from filecmp import dircmp, cmp
from pathlib import Path
from difflib import Differ
import re
import fileinput

import jsbeautifier
from loguru import logger

import tests.tt_metal.tools.profiler.common as common

REPO_PATH = common.get_repo_path()
TT_METAL_PATH = f"{REPO_PATH}/tt_metal"
GOLDEN_OUTPUTS_DIR = f"{TT_METAL_PATH}/third_party/lfs/profiler/tests/golden/device/outputs"
PROFILER_DIR = f"{TT_METAL_PATH}/tools/profiler/"

RE_RANDOM_ID_STRINGS = [r'if \(document.getElementById\("{0}"\)\) {{', r'    Plotly.newPlot\("{0}", \[{{']


def replace_random_id(line):
    for randomIDStr in RE_RANDOM_ID_STRINGS:
        match = re.search(f"^{randomIDStr.format('.*')}$", line)
        if match:
            return randomIDStr.format("random_id_replaced_for_automation").replace("\\", "")
    return line


def filter_device_analysis_data(testOutputFolder):
    testFiles = os.scandir(testOutputFolder)
    for testFile in testFiles:
        if "device_analysis_data.json" in testFile.name:
            testFilePath = f"{testOutputFolder}/{testFile.name}"
            for line in fileinput.input(testFilePath, inplace=True):
                if "deviceInputLog" not in line:
                    print(line, end="")


def beautify_tt_js_blob(testOutputFolder):
    testFiles = os.scandir(testOutputFolder)
    for testFile in testFiles:
        if ".html" in testFile.name:
            testFilePath = f"{testOutputFolder}/{testFile.name}"
            with open(testFilePath) as htmlFile:
                for htmlLine in htmlFile.readlines():
                    if "!function" in htmlLine:
                        jsBlobs = htmlLine.rsplit("script", 2)
                        assert len(jsBlobs) > 2, f"{testFile.name} has more than one JS section"
                        jsBlob = jsBlobs[1].strip(' <>/"')
                        beautyJS = jsbeautifier.beautify(jsBlob)
                        break

            os.remove(testFilePath)

            beautyJS = beautyJS.split("\n")
            jsLines = []
            for jsLine in beautyJS:
                jsLines.append(replace_random_id(jsLine))

            beautyJS = "\n".join(jsLines)

            with open(f"{testFilePath.split('.html')[0]}.js", "w") as jsFile:
                jsFile.writelines(beautyJS)


def run_device_log_compare_golden(test):
    goldenPath = f"{GOLDEN_OUTPUTS_DIR}/{test}"
    underTestPath = f"{PROFILER_DIR}/output"

    ret = os.system(
        f"cd {PROFILER_DIR} && ./process_device_log.py -d {goldenPath}/profile_log_device.csv --no-print-stats --no-artifacts --no-webapp"
    )
    assert ret == 0, f"Log process script crashed with exit code {ret}"

    beautify_tt_js_blob(underTestPath)
    filter_device_analysis_data(underTestPath)

    dcmp = dircmp(goldenPath, underTestPath)

    for diffFile in dcmp.diff_files:
        goldenFile = Path(f"{goldenPath}/{diffFile}")
        underTestFile = Path(f"{underTestPath}/{diffFile}")

        diffStr = f"\n{diffFile}\n"
        with open(goldenFile) as golden, open(underTestFile) as underTest:
            differ = Differ()
            for line in differ.compare(golden.readlines(), underTest.readlines()):
                if line[0] in ["-", "+", "?"]:
                    diffStr += line
        logger.error(diffStr)

    assert not dcmp.diff_files, f"{dcmp.diff_files} cannot be different from golden"
    assert not dcmp.right_only, f"New output files: {dcmp.right_only}"
    assert not dcmp.left_only, f"Golden files not present in output: {dcmp.left_only}"
    assert not dcmp.funny_files, f"Unreadable files: {dcmp.funny_files}"


def run_test(func):
    def test():
        run_device_log_compare_golden(func.__name__)

    return test
