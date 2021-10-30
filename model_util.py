import os
import torch
import config as args


def save_model(model, output_dir,name):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, name)
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(output_dir,name):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir, name)
    model_state_dict = torch.load(output_model_file,map_location='cpu')
    return model_state_dict