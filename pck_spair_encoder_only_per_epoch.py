import os
from types import SimpleNamespace
import pickle as pkl
from pck_spair_encoder_only import main

if __name__ == '__main__':
    EVAL_DIR = os.path.join(
        '/gscratch/krishna/gstoica3/research/MergedVisionEncoders',
        'checkpoints',
        'structural_distillation',
        'rep_cosine-struct_mse-struct_layers_clip_last_4_blocks_dino_last_4_blocks',
        'hyperparams',
        'seed_42-batch_32-epochs_10-lr_0.0001-struct_1.0-rep_1.0only_train_struct_blocks'
    )
    CONFIG = {
        'SAMPLE': 20,
        'ONLY_DINO': True,
        'MODEL_SIZE': 'openclip_vitb16',
        # Defaults
        'MASK': False,
        'DIST': 'l2',
        'SEED': 42,
        'COUNT_INVIS': False,
        'TOTAL_SAVE_RESULT': 5,
        'IMG_THRESHOLD': False,
        'VER': 'v1-5',
        'PROJ_LAYER': False,
        'CO_PCA_DINO': 0,
        'PCA_DIMS': [256, 256, 256],
        'TIMESTEP': 100,
        'SIZE': 960,
        'INDICES': [2,5,8,11],
        'EDGE_PAD': False,
        'WEIGHT': [1,1,1,1,1],
        'RAW': False,
        'NOT_FUSE': False,
        'DINOV1': False,
        'DRAW_DENSE': False,
        'DRAW_SWAP': False,
        'DRAW_GIF': False,
        'TEXT_INPUT': False,
        'PASCAL': False,
        'NOTE': '',
    }
    WRITE_DIR = os.path.join(
        '/gscratch/krishna/gstoica3/research/sd-dino/correspondences_by_epoch',
        'rep_cosine-struct_mse-struct_layers_clip_last_4_blocks_dino_last_4_blocks',
        'seed_42-batch_32-epochs_10-lr_0.0001-struct_1.0-rep_1.0only_train_struct_blocks'
    )
    os.makedirs(WRITE_DIR, exist_ok=True)
    
    for model_fname in sorted(os.listdir(EVAL_DIR)):
        if not model_fname.startswith('epoch'): continue
        if model_fname == 'epoch_2-loss=0.01.pth': continue
        
        model_path = os.path.join(EVAL_DIR, model_fname)
        CONFIG['MODEL_LOAD_PATH'] = model_path
        args = SimpleNamespace(**CONFIG)
        best_pcks = main(args)
        
        with open(os.path.join(WRITE_DIR, f'{model_fname}.pkl'), 'wb') as f:
            pkl.dump(best_pcks, f)