from huggingface_hub import PyTorchModelHubMixin
import vision_transformer

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

def get_feature_extractor(pretrained=True):
    model = neuroFM_HE20x.from_pretrained("MountSinaiCompPath/neuroFM_HE20x") if pretrained else neuroFM_HE20x()
    return model