"""
Builds and returns a neural network model based on the specified arguments.

Args:
    args (argparse.Namespace): The arguments containing the model name (mdname).
    in_channels (int): The number of input channels for the model.
    out_channels (int): The number of output channels for the model.

Returns:
    torch.nn.Module: The constructed neural network model.

Raises:
    ValueError: If the specified model name (mdname) is not supported.

Supported Models:
    - 'UNET': Constructs a 3D U-Net model.
    - 'UNETR': Constructs a U-Net Transformer model.
    - 'SWIN_UNETR': Constructs a Swin U-Net Transformer model.
"""

import os
import torch
from monai.networks.nets import UNet, UNETR, SwinUNETR


def build_model(args, in_channels, out_channels, img_size=[96, 96, 96]):
    """
    Build and initialize a deep learning model based on the specified architecture.

    Args:
        args (Namespace): A namespace object containing model configuration parameters.
                          The `mdname` attribute specifies the model type ('UNET', 'UNETR', or 'SWIN_UNETR').
        in_channels (int): The number of input channels for the model.
        out_channels (int): The number of output channels for the model.
        img_size (list, optional): The spatial dimensions of the input image. Defaults to [96, 96, 96].

    Returns:
        torch.nn.Module: The initialized model moved to the specified device.

    Raises:
        ValueError: If the specified model name (`args.mdname`) is not supported.
    """

    if args.mdname == 'UNET':
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif args.mdname == 'UNETR':
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif args.mdname == 'SWIN_UNETR':
        model = SwinUNETR(
            #img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
        )
    elif args.mdname == 'SWIN_UNETRV2':
        model = SwinUNETR(
            #img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True,
        )
    else:
        raise ValueError(f"Does not support '{args.mdname}' model!")

    
    if args.use_pretrained is True:
        print("Loading Weights from the Path {}".format(args.pretrained_path))
        pretrained_model_dict = torch.load(args.pretrained_path, weights_only=True)
        if "state_dict" in pretrained_model_dict:
            # If the state_dict is wrapped in a dictionary, extract it.
            pretrained_model_dict = pretrained_model_dict["state_dict"]
        else:
            # If the state_dict is not wrapped, use it directly.
            print("No 'state_dict' key found in the pretrained model dictionary. Using it directly.")   
            pretrained_model_weights = pretrained_model_dict
        if args.mdname == 'UNETR':
            # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
            # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
            # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
            # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
            model_dict = model.vit.state_dict()
            model_weights = {k: v for k, v in pretrained_model_weights.items() if k in model_dict}
            model_dict.update(model_weights)
            model.vit.load_state_dict(model_dict)
        elif args.mdname == 'SWIN_UNETR' or args.mdname == 'SWIN_UNETRV2':
            # Remove items of swin_weights if they are not in the Swin Transformer backbone.
            model_dict = model.state_dict()
            model_weights = {k: v for k, v in pretrained_model_weights.items() if k in model_dict}
            model_dict.update(model_weights)
            model.load_state_dict(model_dict)
        else:
            # For UNET, we can directly load the weights.
            model.load_state_dict(pretrained_model_weights)

        del model_dict, pretrained_model_weights, pretrained_model_dict
        
        print("Pretrained Weights Succesfully Loaded !")
    elif args.use_pretrained is False:
        print("No custom weights were loaded, all weights being used are randomly initialized!")

    return model
