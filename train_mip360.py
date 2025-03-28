import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch run RegSegField training on MipNeRF-360 scenes")
    parser.add_argument("--data_root", type=str, required=True, help="Root path containing all MipNeRF-360 scene folders")
    parser.add_argument("--scenes", nargs="+", required=True, help="List of scene names")
    parser.add_argument("--input_views", type=int, default=12)
    parser.add_argument("--downscale_factor", type=int, default=8)
    parser.add_argument("--llff_hold", type=bool, default=True)
    parser.add_argument("--mask_dir", type=str, default="masks_allprompts")
    parser.add_argument("--reg_from", type=int, default=800)
    parser.add_argument("--reg_steps", type=int, default=3500)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--random_scale", type=int, default=15)
    parser.add_argument("--mask_thresh", type=int, default=1)
    parser.add_argument("--background_color", type=str, default="black")
    parser.add_argument("--tv_weight", type=float, default=0.1)

    args = parser.parse_args()

    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "regseg-splatfacto=regseg_splatfacto.sam_gaussian_config:sam_group_gaussian_config"
    os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "sam-dataparser=regseg_datasets.SAM_dataparser_config:sam_dataparser"

    for scene in args.scenes:
        for use_reg in [True, False]:
            random_init = False
            use_depth = False  # as in original bash script

            experiment_name = f"{scene}_reg{use_reg}_depth{use_depth}_random{random_init}_001reg"

            command = [
                "python", "-m", "nerfstudio.scripts.train", "regseg-splatfacto",
                "--data", os.path.join(args.data_root, scene),
                "--experiment_name", experiment_name,
                "--pipeline.use-reg", str(use_reg),
                "--pipeline.model.use_depth", str(use_depth),
                "--pipeline.model.random_init", str(random_init),
                "--pipeline.model.random_scale", str(args.random_scale),
                "--pipeline.model.mask_count_thresh", str(args.mask_thresh),
                "--pipeline.model.use_binary_mask_for_training", str(random_init),
                "--pipeline.reg_from", str(args.reg_from),
                "--pipeline.reg_steps", str(args.reg_steps),
                "--pipeline.reg_weight", str(args.reg_weight),
                "--pipeline.learn_grouping", "True",
                "--pipeline.train_mask_encoder", "True",
                "--pipeline.model.background_color", args.background_color,
                "--pipeline.use_tv_loss", "True",
                "--pipeline.tv_weight", str(args.tv_weight),
                "--vis", "wandb",
                "sam-dataparser",
                "--use_mask_matching", "True",
                "--mask_loader_ckpt", f"{scene}_{args.input_views}_sorted.pth",
                "--use_llff_hold", str(args.llff_hold),
                "--input_views", str(args.input_views),
                "--downscale_factor", str(args.downscale_factor),
                "--masks_path", args.mask_dir,
                "--use_point_from_image", str(random_init),
                "--sort_by_name", "True"
            ]

            print(f"\n==> Running: {experiment_name}")
            subprocess.run(command)

if __name__ == "__main__":
    main()
