import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch run RegSegField training across scenes")
    parser.add_argument("--data_root", type=str, required=True, help="Root path containing all scene folders")
    parser.add_argument("--scenes", nargs="+", required=True, help="List of scene names")
    parser.add_argument("--input_views", type=int, default=3)
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--llff_hold", type=bool, default=False)
    parser.add_argument("--mask_dir", type=str, default="masks_allprompts")
    parser.add_argument("--binary_mask_dir", type=str, default="idr_masks")
    parser.add_argument("--reg_from", type=int, default=800)
    parser.add_argument("--reg_weight", type=float, default=0.05)
    parser.add_argument("--random_scale", type=int, default=15)
    parser.add_argument("--mask_thresh", type=int, default=2)
    parser.add_argument("--background_color", type=str, default="black")

    args = parser.parse_args()

    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "regseg-splatfacto=regseg_splatfacto.sam_gaussian_config:sam_group_gaussian_config"
    os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "sam-dataparser=regseg_datasets.SAM_dataparser_config:sam_dataparser"

    for scene in args.scenes:
        for use_reg in [True, False]:
            for use_depth in [True, False]:
                random_init = False
                experiment_name = f"{scene}_reg{use_reg}_depth{use_depth}_random{random_init}_metric"

                command = [
                    "python", "-m", "nerfstudio.scripts.train", "regseg-splatfacto",
                    "--data", os.path.join(args.data_root, scene),
                    "--experiment_name", experiment_name,
                    "--pipeline.use-reg", str(use_reg),
                    "--pipeline.use_depth", str(use_depth),
                    "--pipeline.model.random_init", str(random_init),
                    "--pipeline.model.random_scale", str(args.random_scale),
                    "--pipeline.model.mask_count_thresh", str(args.mask_thresh),
                    "--pipeline.model.use_binary_mask_for_training", str(random_init),
                    "--pipeline.reg_from", str(args.reg_from),
                    "--pipeline.reg_weight", str(args.reg_weight),
                    "--pipeline.learn_grouping", "True",
                    "--pipeline.train_mask_encoder", "True",
                    "--pipeline.model.background_color", args.background_color,
                    "--pipeline.use_tv_loss", "True",
                    "--vis", "wandb",
                    "sam-dataparser",
                    "--use_mask_matching", "True",
                    "--mask_loader_ckpt", f"{scene}_{args.input_views}_sorted.pth",
                    "--use_llff_hold", str(args.llff_hold),
                    "--input_views", str(args.input_views),
                    "--downscale_factor", str(args.downscale_factor),
                    "--masks_path", args.mask_dir,
                    "--use_point_from_image", str(random_init),
                    "--binary_masks_path", args.binary_mask_dir
                ]

                print(f"\n==> Running: {experiment_name}")
                subprocess.run(command)

if __name__ == "__main__":
    main()
