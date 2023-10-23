import os
import pickle

from torchvision.utils import save_image

from train.encode import load_first_stage


def sample(denoisez_path, first_stage_model, conf, output):
    """
    sample from denoisez
    :param denoisez_path: denoisez dataset path
    :param first_stage_model: model state dict path of first stage
    :param conf: first stage config
    :param output: output path of sampled results
    :return:
    """
    autoencoder, _ = load_first_stage(conf, first_stage_model, False)
    for i in range(20):
        pkl_path = f"{denoisez_path}/{i}.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                denoisez_dataset = pickle.load(f)
                images = autoencoder._module.decode(denoisez_dataset)
                os.makedirs(output, exist_ok=True)
                for j, image in enumerate(images):
                    save_image(image,  conf.output + f'/{i * 300 + j}.png')