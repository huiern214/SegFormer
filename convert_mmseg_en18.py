import os
import shutil
import os.path as osp
from tqdm import tqdm

def convert_endovis18_pre_split(
        src_root="./data/en17to18_thres0.5/endovis_2018/fold0",
        save_root="./data/mmseg_en17to18_thres0.5/fold0"):

    # Define fixed subdirs (no folds)
    sets = ["train", "val"]

    # Create MMSeg format structure
    for sub in ["img_dir/train", "img_dir/val", "ann_dir/train", "ann_dir/val", "splits"]:
        os.makedirs(osp.join(save_root, sub), exist_ok=True)

    train_txt = open(osp.join(save_root, "splits", "train.txt"), "w")
    val_txt   = open(osp.join(save_root, "splits", "val.txt"), "w")

    for split in sets:
        print(f"\n====== Processing {split} set ======")

        if split == "train":
            src_img_root = osp.join(src_root, split, "0", "images")
            src_ann_root = osp.join(src_root, split, "0", "annotations")
        else:
            src_img_root = osp.join(src_root, split, "images")
            src_ann_root = osp.join(src_root, split, "annotations")

        if not osp.exists(src_img_root):
            print(f"Missing folder: {src_img_root} - skipping")
            continue

        # Loop through each sequence folder (seq1, seq3, etc.)
        for seq in sorted(os.listdir(src_img_root)):
            seq_img_dir = osp.join(src_img_root, seq)
            seq_ann_dir = osp.join(src_ann_root, seq)

            if not osp.isdir(seq_img_dir):
                continue

            files = sorted(os.listdir(seq_img_dir))

            for img_file in tqdm(files, desc=f"{split}/{seq}"):
                base_name = img_file.split(".")[0]
                base = f"{seq}_{base_name}"

                img_src  = osp.join(seq_img_dir, img_file)
                mask_src = osp.join(seq_ann_dir, img_file.replace(".jpg", ".png"))

                if not osp.exists(mask_src):
                    print(f"Missing mask: {mask_src} - skipping")
                    continue

                img_dst  = osp.join(save_root, f"img_dir/{split}",  f"{base}.jpg")
                mask_dst = osp.join(save_root, f"ann_dir/{split}", f"{base}.png")

                shutil.copy(img_src, img_dst)
                shutil.copy(mask_src, mask_dst)

                if split == "train":
                    train_txt.write(f"{base}\n")
                else:
                    val_txt.write(f"{base}\n")

    train_txt.close()
    val_txt.close()
    print(f"\nConversion complete! Output saved to: {save_root}")


if __name__ == "__main__":
    convert_endovis18_pre_split(
        src_root="./data/endovis_2018",
        save_root="./data/mmseg_endovis_2018")
