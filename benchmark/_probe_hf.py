import os
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
from huggingface_hub import get_hf_file_metadata, hf_hub_url
url = hf_hub_url("ai4bharat/MSMARCO-XI", "validation/telval.parquet", repo_type="dataset")
meta = get_hf_file_metadata(url)
print("hf ok size", meta.size)
