#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

main_module_path = "torchview"

for path in sorted(Path(main_module_path).rglob("*.py")):
    print(f"Processing {path}")
    module_path = path.relative_to(main_module_path).with_suffix("")
    doc_path = path.relative_to(main_module_path).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        # parts = parts[:-1]
        continue
    elif parts[-1] == "__version__":
        continue
    elif parts[-1] == "__main__":
        continue
    elif parts[-1] == " ":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w+") as fd:
        identifier = ".".join(parts)
        print(f"::: torchview.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

#with mkdocs_gen_files.open("reference/SUMMARY.md", "w+") as nav_file:
#    nav_file.writelines(nav.build_literate_nav())