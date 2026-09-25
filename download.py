from huggingface_hub import snapshot_download

print("Starting download of baochenfu/OCT-Bench dataset from Hugging Face...")
snapshot_download(
    repo_id="baochenfu/OCT-Bench",
    repo_type="dataset",
    local_dir="C:/Users/HP/Desktop/OCT-Bench"
)

print("Download completed!")
