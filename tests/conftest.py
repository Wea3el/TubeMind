"""Configure the test environment before any tubemind module is imported.

tubemind.config calls load_environment() at module level, which raises if
OPENAI_API_KEY or YOUTUBE_API_KEY are missing. Setting dummy values here
(before any import of tubemind) prevents that raise during test collection.

TUBEMIND_DATA_DIR is pointed at a temp directory so the test DB never touches
the real .local/tubemind.db that the running app uses.
"""

import os
import tempfile

_tmp_data_dir = tempfile.mkdtemp(prefix="tubemind_test_")

os.environ.setdefault("OPENAI_API_KEY", "test-key-unit-tests")
os.environ.setdefault("YOUTUBE_API_KEY", "test-key-unit-tests")
os.environ.setdefault("OPENAI_MODEL", "gpt-4.1-nano")
os.environ["TUBEMIND_DATA_DIR"] = _tmp_data_dir
