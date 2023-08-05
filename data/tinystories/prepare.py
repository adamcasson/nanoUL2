import h5py
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories")

    enc = AutoTokenizer.from_pretrained("../../ul2-tinystories-tokenizer")

    def process(example):
        ids = enc.encode(example["text"])
        # ids.append(enc.eos_token_id) # add the end of text token
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        # Open HDF5 file and create dataset
        f = h5py.File(f"{split}.h5", "w")
        dt = h5py.special_dtype(vlen=np.uint16)
        h5dset = f.create_dataset(split, (len(dset),), dtype=dt)

        # Save each array in the list as a variable length element
        for i in tqdm(range(len(dset)), desc=f"writing {split}.h5"):
            h5dset[i] = np.asarray(dset[i]["ids"], dtype=np.uint16)

        f.close()
