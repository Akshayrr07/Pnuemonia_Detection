def load_all_models(device):
    import torch
    from src.models.model_factory import get_model

    model_names = ["mobilenet", "efficientnet", "resnet"]

    models = []

    for name in model_names:
        model = get_model(name, freeze=False).to(device)

        model_path = f"saved_models/{name}.pt"   # ✅ ensure STRING

        state_dict = torch.load(model_path, map_location=device)  # ✅ SAFE LOAD
        model.load_state_dict(state_dict)

        model.eval()
        models.append(model)

    return models