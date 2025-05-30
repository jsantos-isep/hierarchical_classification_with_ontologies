import torch

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def device_1():
    return 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU



def freeze_all_layers_but_classifier(model):
    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'classification' not in name:  # If the parameter is not part of the classifier
            param.requires_grad = False  # Freeze it

def process_json(data, flag):
    if flag == "top":
        return list(data.keys())  # Return only the keys
    else:
        return [item for sublist in data.values() for item in sublist]  # Flatten values
