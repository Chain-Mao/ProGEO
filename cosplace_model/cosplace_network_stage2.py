
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple

from cosplace_model.layers import Flatten, L2Norm, GeM

import parser
args = parser.parse_arguments()

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "CLIP-RN50": 2048,
    "CLIP-RN101": 2048,
    "CLIP-ViT-B-16": 512,
    "CLIP-ViT-B-32": 512,
}

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone_name : str, fc_output_dim : int, train_all_layers : bool = False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            train_all_layers (bool): whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone_name in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone_name = backbone_name
        clip_model, features_dim = get_backbone(backbone_name, train_all_layers)
        self.backbone = clip_model.visual
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)
        if "RN" in backbone_name:
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )

    def forward(self, x=None, prompt_learner=None, label=None, get_text=False):
        if get_text == True:
            prompts = prompt_learner(label)
            text_features = self.text_encoder(prompts, prompt_learner.tokenized_prompts)
            return text_features
        if self.backbone_name == "CLIP-RN50":
            image_features = self.backbone(x, stage=2)
            image_features = self.aggregation(image_features)
        elif self.backbone_name == "CLIP-RN101":
            image_features = self.backbone(x, stage=2)
            image_features = self.aggregation(image_features)
        elif self.backbone_name == "CLIP-ViT-B-16":
            image_features = self.backbone(x)
        elif self.backbone_name == "CLIP-ViT-B-32":
            image_features = self.backbone(x)
        return image_features


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    # # CLIP模型导入方式不同
    # h_resolution = int(([512, 512][0] - 16) // [16, 16][0] + 1)
    # w_resolution = int(([512, 512][1] - 16) // [16, 16][1] + 1)
    # vision_stride_size = 16
    if backbone_name == "CLIP-RN50":
        clip_model = load_clip_to_cpu("RN50")
        clip_model.to("cuda")
        return clip_model

    elif backbone_name == "CLIP-RN101":
        clip_model = load_clip_to_cpu("RN101")
        clip_model.to("cuda")
        return clip_model

    elif backbone_name == "CLIP-ViT-B-16":
        clip_model = load_clip_to_cpu("ViT-B/16")
        clip_model.to("cuda")
        return clip_model

    elif backbone_name == "CLIP-ViT-B-32":
        clip_model = load_clip_to_cpu("ViT-B/32")
        clip_model.to("cuda")
        return clip_model

    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str, train_all_layers : bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if "ViT" in backbone_name:
        if train_all_layers:
            # 冻结除了图像编码器以外的所有层
            for param in backbone.parameters():
                param.requires_grad = False
            # 解冻图像编码器层
            for param in backbone.visual.parameters():
                param.requires_grad = True
            logging.debug("Train all visual layers of the CLIP")
        else:
            # 冻结除了图像编码器以外的所有层
            for param in backbone.parameters():
                param.requires_grad = False
            # 解冻图像编码器层
            for param in backbone.visual.parameters():
                param.requires_grad = True
            # 冻结图像编码器的conv1和ln_pre
            for param in backbone.visual.conv1.parameters():
                param.requires_grad = False
            for param in backbone.visual.ln_pre.parameters():
                param.requires_grad = False
            # 冻结图像编码器的前N层
            for i, layer in enumerate(backbone.visual.transformer.resblocks):
                # if i < 6:
                if i < args.freeze_trans:
                    for param in layer.parameters():
                        param.requires_grad = False
            logging.debug("Train last some layers of the CLIP visual, freeze others")

    elif "RN" in backbone_name:
        if train_all_layers:
            # 冻结除了图像编码器以外的所有层
            for param in backbone.parameters():
                param.requires_grad = False
            # 解冻图像编码器层
            for param in backbone.visual.parameters():
                param.requires_grad = True
            logging.debug("Train all visual layers of the CLIP")
        else:
            # 冻结除了图像编码器以外的所有层
            for param in backbone.parameters():
                param.requires_grad = False
            # 解冻图像编码器层
            for param in backbone.visual.parameters():
                param.requires_grad = True
            # 冻结图像编码器的前几层
            for name, child in backbone.visual.named_children():
                # if name == "layer3":  # Freeze layers before conv_3
                if name == "layer" + str(args.freeze_cnn+1):
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        # layers = backbone.visual
        # layers = list(backbone.visual.children())[:-1]    # 直接使用源模型效果不如改进后的forward好

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    return backbone, features_dim

from clip import clip
def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    model = clip.build_model(state_dict or model.state_dict(), image_size=512)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X street."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors.cuda())

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts