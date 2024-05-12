
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from cosplace_model import cosplace_network_stage2
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
model = cosplace_network_stage2.GeoLocalizationNet(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)

test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold, resize_test_imgs=args.resize_test_imgs)

recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

'''
CUDA_VISIBLE_DEVICES=0 python3 eval.py --backbone CLIP-ViT-B-16 --resume_model /data1/CosPlace/logs/default/stage2/VIT16_nofreeze/best_model.pth --test_set_folder /data3/VPR-datasets-downloader/msls/val --resize_test_imgs --infer_batch_size 128 --fc_output_dim 512
CUDA_VISIBLE_DEVICES=1 python3 eval.py --backbone CLIP-ViT-B-16 --resume_model /data1/CosPlace/logs/default/stage2/VIT16_nofreeze_triplet/best_model.pth --test_set_folder /data3/VPR-datasets-downloader/msls/val --resize_test_imgs --infer_batch_size 128 --fc_output_dim 512

CUDA_VISIBLE_DEVICES=2 python3 eval.py --backbone CLIP-RN50 --resume_model /data1/CosPlace/logs/default/stage2/RN50_nofreeze/best_model.pth --test_set_folder /data3/VPR-datasets-downloader/msls/val --infer_batch_size 128 --fc_output_dim 1024
CUDA_VISIBLE_DEVICES=3 python3 eval.py --backbone CLIP-RN50 --resume_model /data1/CosPlace/logs/default/stage2/RN50_nofreeze_triplet/best_model.pth --test_set_folder /data3/VPR-datasets-downloader/msls/val --infer_batch_size 128 --fc_output_dim 1024
'''
