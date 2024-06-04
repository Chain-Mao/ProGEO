import os
import shutil
import sys
import torch
import logging
import numpy as np
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import util
import parser
import commons
from cosplace_model import cosplace_network_stage1
from datasets.train_dataset import TrainDatasetStage1
from cosplace_model.supcontrast import SupConLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from cosplace_model.cosplace_network_stage1 import PromptLearner

torch.backends.cudnn.benchmark = True  # Provides a speedup

# 删除cache文件夹
cache_path = "./cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/stage1/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Datasets
groups = [TrainDatasetStage1(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                             current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

#### Model
model = cosplace_network_stage1.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)

prompt_learners = [PromptLearner(group.get_classes_num(), model.dtype, model.token_embedding) for group in groups]
for name, param in prompt_learners[0].named_parameters():
    if param.requires_grad:
        print(f"模型可更新参数: {name}, 维度: {param.shape}")
        
prompt_optimizers = [torch.optim.Adam(prompt_learner.parameters(), lr=args.lr_stage1) for prompt_learner in prompt_learners]
schedulers = [CosineAnnealingLR(prompt_optimizer, T_max=args.epochs_num_stage1//len(groups)) for prompt_optimizer in prompt_optimizers]    # 余弦退火学习率衰减

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model_stage1 is not None:
    logging.debug(f"Loading model from {args.resume_model_stage1}")
    model_state_dict = torch.load(args.resume_model_stage1)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {groups[0].get_classes_num()} classes and {groups[0].get_images_num()} images for the first group, " +
             f"with batch_size {args.batch_size_stage1}")

if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            T.Resize([224, 224]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

scaler = torch.cuda.amp.GradScaler()    # 自动混合精度
xent = SupConLoss(args.device)

# 第一阶段训练
logging.info('start training stage1')
if args.cache_feature_folder and os.listdir(args.cache_feature_folder):   # lists have been saved in cache feature folder
    image_features_list_all = torch.load(os.path.join(args.cache_feature_folder, "image_features_list_all.pth"))
    labels_list_all = torch.load(os.path.join(args.cache_feature_folder, "labels_list_all.pth"))
    print("lists have been saved in cache feature folder")
else:
    image_features_list_all = []
    labels_list_all = []
    for g in range(len(groups)):
        image_features = []
        labels = []
        dataloader = torch.utils.data.DataLoader(groups[g], num_workers=args.num_workers, batch_size=args.batch_size_stage1, shuffle=True, pin_memory=(args.device == "cuda"))
        with torch.no_grad():
            for images, targets, _ in tqdm(dataloader, ncols=100):
                images = images.to(args.device)
                targets = targets.to(args.device)
                if args.augmentation_device == "cuda":
                    images = gpu_augmentation(images)
                with torch.cuda.amp.autocast(enabled=True):
                    image_feature = model(images)
                    for i, img_feat in zip(targets, image_feature):
                        labels.append(i)
                        image_features.append(img_feat.cpu())
            labels_list = torch.stack(labels, dim=0)
            image_features_list = torch.stack(image_features, dim=0)
        labels_list_all.append(labels_list)
        image_features_list_all.append(image_features_list)
        del labels, image_features, labels_list, image_features_list
        logging.info(f"Group {g} images feature have been extracted")
    # torch.save(image_features_list_all, os.path.join(args.cache_feature_folder, "image_features_list_all.pth"))
    # torch.save(labels_list_all, os.path.join(args.cache_feature_folder, "labels_list_all.pth"))

for epoch_num in range(0, args.epochs_num_stage1):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num

    prompt_learners[current_group_num] = prompt_learners[current_group_num].to(args.device)
    util.move_to_device(prompt_optimizers[current_group_num], args.device)

    labels_list = labels_list_all[current_group_num].cuda()
    image_features_list = image_features_list_all[current_group_num].cuda()
    model = model.train()

    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    batch = args.batch_size_stage1
    num_image = labels_list.shape[0]
    i_ter = num_image // batch
    iter_list = torch.randperm(num_image).to(args.device)
    for i in tqdm(range(i_ter + 1), ncols=100):
        prompt_optimizers[current_group_num].zero_grad()
        if i != i_ter:
            b_list = iter_list[i * batch:(i + 1) * batch]
        else:
            b_list = iter_list[i * batch:num_image]

        target = labels_list[b_list]
        image_features = image_features_list[b_list]
        with torch.cuda.amp.autocast(enabled=True):
            text_features = model(prompt_learner=prompt_learners[current_group_num], label=target, get_text=True)
        loss_i2t = xent(image_features, text_features, target, target)
        loss_t2i = xent(text_features, image_features, target, target)
        loss = loss_i2t + loss_t2i

        scaler.scale(loss).backward()
        scaler.step(prompt_optimizers[current_group_num])
        scaler.update()
        epoch_losses = np.append(epoch_losses, loss.item())

    schedulers[current_group_num].step()

    prompt_learners[current_group_num] = prompt_learners[current_group_num].cpu()
    util.move_to_device(prompt_optimizers[current_group_num], "cpu")
    
    logging.debug("Stage1 Epoch %02d in %s, loss = %.4f, learning rate = %.2e",
                  epoch_num, str(datetime.now() - epoch_start_time)[:-7],
                  epoch_losses.mean(), schedulers[current_group_num].get_last_lr()[0])

    if (epoch_num+1) % args.checkpoint_period_stage1 == 0:
        torch.save(prompt_learners, f"{args.output_folder}/prompt_learners_{epoch_num+1}.pth")
torch.save(prompt_learners, f"{args.output_folder}/last_prompt_learners.pth")

logging.info(f"Trained for stage1 {args.epochs_num_stage1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
