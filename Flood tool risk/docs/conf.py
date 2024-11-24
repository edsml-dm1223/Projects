import sys
import os

project_path = os.getcwd()

sys.path.insert(0, os.path.abspath(os.path.join(os.curdir, project_path)))

project = "Flood Tool"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autoclass_content = "both"