# see the following links for version compatitsli
# // https://github.com/pytorch/text
# https://github.com/pytorch/vision

from __future__ import annotations

from typing import Any, Callable, Union


def get_version_dict(file_path: str) -> Any:
    # reads json file and returns the version number
    import json
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
def flatten_dict(d: dict[str, Any]) -> list[dict[str, str]]:
    x = []
    for k, v in d.items():
        for (torch_v, torchvision_v, torchtext_v) in v:
            x.append({'python_version': k, 'torch_version': torch_v, 'torchvision_version': torchvision_v, 'torchtext_version': torchtext_v})
    return x

if __name__ == '__main__':
    import argparse
    # take single cli integer argument
    parser = argparse.ArgumentParser(description="This script takes a single integer argument")
    parser.add_argument("integer", type=int, help="an integer")
    # add option to print numpy or python version
    parser.add_argument("--variable-name", type=str, help="python, torch, torchvision, torchtext", default="python")
    args = parser.parse_args()
    version_id = args.integer

    file_path = 'scripts/version_map.json'
    version_dict = get_version_dict(file_path)
    version_list = flatten_dict(version_dict)
    # print(f"version with id: {version_id} is {version_list[version_id]}")
    variable_name = args.variable_name
    if variable_name == "python":
        print(f"{version_list[version_id]['python_version']}")
    elif variable_name == "torch":
        print(f"{version_list[version_id]['torch_version']}")
    elif variable_name == "torchvision":
        print(f"{version_list[version_id]['torchvision_version']}")
    elif variable_name == "torchtext":
        print(f"{version_list[version_id]['torchtext_version']}")
    elif variable_name == "ids":
        print(f"{[i for i in range(len(version_list))]}")
    else:
        raise ValueError("variable-name should be numpy_version or python_version")