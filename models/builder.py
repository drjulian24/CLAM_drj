import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
##below added by DRJ 05/12/2025
import vision_transformer
import nn.Module
from huggingface_hub import PyTorchModelHubMixin


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

#neuro_FM class added by DRJ 05/12/2025. As taken from Github documentation of model
class neuroFM_HE20x(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer='swiglufused',
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        self.encoder = vision_transformer.__dict__['vit_large'](**vit_kwargs)
    
    def forward(self, x):
        return self.encoder(x)
    
#neuroFM check added by DRJ 05/12/2025
def has_neuroFM_HE20x():
    try:
        # Test loading the model (optional, for validation)
        neuroFM_HE20x.from_pretrained("MountSinaiCompPath/neuroFM_HE20x")
        ckpt_path = None  # Use None for Hugging Face download
        return True, ckpt_path
    except Exception as e:
        print(f"neuroFM_HE20x not available: {e}")
        return False, None

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    #neuroFM info added by DRJ 05/12/2025
    elif model_name == 'neuroFM_HE20x':
        HAS_NEUROFM, NEUROFM_CKPT_PATH = has_neuroFM_HE20x()
        assert HAS_NEUROFM, 'neuroFM_HE20x is not available'
        if NEUROFM_CKPT_PATH:
            model = neuroFM_HE20x()  # Initialize model
            state_dict = torch.load(NEUROFM_CKPT_PATH)
            model.load_state_dict(state_dict)
        else:
            model = neuroFM_HE20x.from_pretrained("MountSinaiCompPath/neuroFM_HE20x")
        img_transforms = transforms.Compose([
            transforms.Resize((target_img_size, target_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        model.eval()

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms