import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import os
import shutil
import test
import util
import parser
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network_stage2
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from cosplace_model.softmax_loss import CrossEntropyLabelSmooth
import triplet_loss

torch.backends.cudnn.benchmark = True  # Provides a speedup

# 删除cache文件夹
cache_path = "./cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/stage2/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold,
                     image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold,
                      image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Model
model = cosplace_network_stage2.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"模型可更新参数: {name}, 维度: {param.shape}")
        
prompt_learners = torch.load(args.prompt_learners)
prompt_learners = [prompt_learner.cuda() for prompt_learner in prompt_learners]

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
# model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
        augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                contrast=args.contrast,
                                                saturation=args.saturation,
                                                hue=args.hue),
        augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                      scale=[1 - args.random_resized_crop, 1]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

# 第二阶段训练
logging.info('start training stage2')
text_features_list_all = []
for g in range(len(groups)):
    batch = args.batch_size
    num_classes = len(groups[g])
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            text_feature = model(prompt_learner=prompt_learners[g], label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0)
    text_features_list_all.append(text_features)
del prompt_learners # delete useless to clear memory

for epoch_num in range(start_epoch_num, args.epochs_num):

    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    text_features_list = text_features_list_all[current_group_num].cuda()

    loss_cross = CrossEntropyLabelSmooth(num_classes=len(groups[current_group_num]))

    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)

    dataloader_iterator = iter(dataloader)
    model = model.train()

    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _ = next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)

        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)

        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        if not args.use_amp16:
            image_features = model(images)
            output = classifiers[current_group_num](image_features, targets)
            loss = criterion(output, targets)
            # 添加对比学习损失
            logits = image_features @ text_features_list.t()
            i2tloss = loss_cross(logits, targets)
            loss = loss + i2tloss
            if args.soft_triplet:
                tripletloss = triplet_loss.triplet_loss(image_features, targets)*args.batch_size
                loss = loss + tripletloss

            loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                image_features = model(images)
                output = classifiers[current_group_num](image_features, targets)
                loss = criterion(output, targets)
                # 添加对比学习损失
                logits = image_features @ text_features_list.t()
                i2tloss = loss_cross(logits, targets)
                loss = loss + i2tloss
                if args.soft_triplet:
                    tripletloss = triplet_loss.triplet_loss(image_features, targets) * args.batch_size
                    loss = loss + tripletloss

            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    text_features_list.cpu()

    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")

    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(
        f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)

    if (epoch_num+1) % args.checkpoint_period_stage2 == 0:
        torch.save(model.state_dict(), f"{args.output_folder}/model_{epoch_num+1}.pth")

logging.info(f"Trained for stage2 {args.epochs_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")