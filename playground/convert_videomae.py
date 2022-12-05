import torch


vidoe_mae_checkpoint = torch.load("/path/to/your/video-mae-model.pth")


converted_dict = {"model_state": {}}

for name, p in vidoe_mae_checkpoint["model"].items():
    key = key.replace("encoder.", "backbone.")
    converted_dict['model_state'][key] = p

torch.save(converted_dict, "/path/to/your/video-mae-model-converted.pth")


