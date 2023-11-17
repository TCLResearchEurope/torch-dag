import json

import timm
import tqdm

MODEL_SIZE_JSON = "model_size.json"


def compute_num_params(model: str):
    model = timm.create_model(model, pretrained=False)
    return sum(p.numel() for p in model.parameters()) / 1e6


all_timm_models = timm.list_models('*')

with open(MODEL_SIZE_JSON, "r") as f:
    model_size_dict = json.load(f)
print(f'Current size of the dict: {len(model_size_dict)}')

for id in tqdm.trange(len(all_timm_models)):
    model = all_timm_models[id]
    if model not in model_size_dict:
        model_size_dict[model] = compute_num_params(model)

with open(MODEL_SIZE_JSON, "w+") as f:
    models = json.dump(model_size_dict, f, indent=4, sort_keys=True)
