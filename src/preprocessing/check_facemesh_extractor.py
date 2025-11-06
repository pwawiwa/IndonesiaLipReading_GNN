import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def add_label_to_sample(sample, folder_name, word2id):
    """Add label based on folder name."""
    sample["label"] = word2id[folder_name]
    return sample

def process_pt_file(pt_file, folder_name, word2id, save_every=12000):
    """Load .pt, add labels, save back (in-place)."""
    data = torch.load(pt_file, weights_only=False)
    updated = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(add_label_to_sample, s, folder_name, word2id): s for s in data}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                       desc=f"Labeling {pt_file.name}")):
            sample = future.result()
            updated.append(sample)

            if (i + 1) % save_every == 0:
                torch.save(updated + data[i+1:], pt_file)
    
    torch.save(updated, pt_file)
    return len(updated)

def main():
    base_dir = Path("data/processed")
    word_folders = [p for p in base_dir.iterdir() if p.is_dir()]

    # Build word -> integer label
    word2id = {wf.name: i for i, wf in enumerate(sorted(word_folders))}
    print("Word mapping:", word2id)

    for wf in word_folders:
        for split in ["train.pt", "val.pt", "test.pt"]:
            pt_file = wf / split
            if pt_file.exists():
                print(f"\nProcessing {pt_file}")
                n = process_pt_file(pt_file, wf.name, word2id)
                print(f"✅ Labeled {n} samples in {pt_file}")
            else:
                print(f"⚠️ Missing {pt_file}, skipping.")

if __name__ == "__main__":
    main()
