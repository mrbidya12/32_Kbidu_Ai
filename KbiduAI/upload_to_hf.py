# upload_to_hf.py
from huggingface_hub import HfApi
import os

# The token is automatically picked up from the environment
# (because you already logged in with `hf auth login`)
api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="razz-123/kbidu-30b",
    repo_type="model",
    private=True,
)
print("✅ Kbidu AI uploaded to Hugging Face!")