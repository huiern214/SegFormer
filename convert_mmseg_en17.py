import os
import shutil
import os.path as osp
from tqdm import tqdm

all_seqs = list(range(1, 9))
fold_val = {0: [1, 3], 1: [2, 5], 2: [4, 8], 3: [6, 7]}

def convert_endovis17(
        src_root="./data/endovis_2017/0",
        save_root="./data/mmseg_endovis_2017",
        fold=0):

    seq_val = fold_val[fold]
    seq_train = [s for s in all_seqs if s not in seq_val]

    print(f"\n====== Fold {fold} ======")
    print("Val seq:", seq_val)
    print("Train seq:", seq_train)

    # MMSeg folder structure
    for sub in ["img_dir/train", "img_dir/val", "ann_dir/train", "ann_dir/val", "splits"]:
        os.makedirs(osp.join(save_root, sub), exist_ok=True)

    train_txt = open(osp.join(save_root, "splits", f"fold{fold}_train.txt"), "w")
    val_txt   = open(osp.join(save_root, "splits", f"fold{fold}_val.txt"), "w")

    for seq in tqdm(all_seqs, desc=f"Processing fold{fold}"):
        src_img_dir = osp.join(src_root, "images", f"seq{seq}")
        src_ann_dir = osp.join(src_root, "annotations", f"seq{seq}")

        if not osp.exists(src_img_dir):
            print(f"Missing folder: {src_img_dir} - skipping seq{seq}")
            continue

        files = sorted(os.listdir(src_img_dir))

        for img_file in files:
            base_name = img_file.split(".")[0]  # e.g. 00034
            base = f"seq{seq}_{base_name}"

            img_src  = osp.join(src_img_dir, img_file)
            mask_src = osp.join(src_ann_dir, img_file.replace(".jpg", ".png"))

            # train or val?
            if seq in seq_val:
                img_dst  = osp.join(save_root, "img_dir/val",  f"{base}.jpg")
                mask_dst = osp.join(save_root, "ann_dir/val", f"{base}.png")
                val_txt.write(f"{base}\n")
            else:
                img_dst  = osp.join(save_root, "img_dir/train",  f"{base}.jpg")
                mask_dst = osp.join(save_root, "ann_dir/train", f"{base}.png")
                train_txt.write(f"{base}\n")

            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)

    train_txt.close()
    val_txt.close()
    print(f"Completed fold{fold}! Output: {save_root}")


if __name__ == "__main__":
    for f in range(4):
        convert_endovis17(fold=f)

    print("\nAll EndoVis 2017 folds converted successfully!")
