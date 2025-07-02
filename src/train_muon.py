from copy import deepcopy
from optimizer.muon import MuonWithAuxAdam
from accelerate import Accelerator


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def creat_model():
    pass

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
        
        
def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    model = creat_model()
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)


    muon_groups = [param for name, param in model.named_parameters() 
                    if param.ndim >= 2 and "pos_embed" not in name and "final_layer" not in name and 
                    "x_embedder" not in name and "t_embedder" not in name and "y_embedder" not in name
                    and param.requires_grad]
    adamw_groups = [param for name, param in model.named_parameters() 
                    if not (param.ndim >= 2 and "pos_embed" not in name and "final_layer" not in name and 
                            "x_embedder" not in name and "t_embedder" not in name and "y_embedder" not in name)
                            and param.requires_grad]

    opt = MuonWithAuxAdam([
            {"params": muon_groups, "use_muon": True, "lr": 1e-3, "weight_decay": 0},
            {"params": adamw_groups, "use_muon": False, "lr": 1e-4, "weight_decay": 0}
        ])

    loader = None


    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)