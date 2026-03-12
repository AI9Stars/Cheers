from .cheers import Cheers_eval

model_name_dict = {
    "cheers": Cheers_eval,
}


def build_model(model_name, alpha=None, cfg=None, steps=None, model_path=None):
    if model_name == "cheers":
        return model_name_dict[model_name](model_path=model_path, alpha=alpha, cfg=cfg, steps=steps)
    else:
        return model_name_dict[model_name]()