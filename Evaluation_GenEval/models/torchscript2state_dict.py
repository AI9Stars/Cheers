import torch

script_model = torch.jit.load("models/ViT-L-14.pt")
state_dict = script_model.state_dict()
# 过滤掉非参数的 key
for bad_key in ["input_resolution", "context_length", "vocab_size"]:
    if bad_key in state_dict:
        print(f"删除无关参数: {bad_key}")
        state_dict.pop(bad_key)

torch.save(state_dict, "models/ViT-L-14-state.pt")