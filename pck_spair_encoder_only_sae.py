import os
import pdb
import sys
import torch
torch.set_num_threads(16)
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import json
from glob import glob
from utils.utils_correspondence import pairwise_sim, draw_correspondences_gathered, chunk_cosine_sim, co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace, draw_correspondences_lines
import matplotlib.pyplot as plt
import sys
import time
from utils.logger import get_logger
from loguru import logger
import argparse
from extractor_dino import ViTExtractor
from extractor_hf_dino import ViTExtractor as ViTExtractorHF
from extractor_hf_openclip import ViTExtractor as OpenClipExtractorHF

from extractor_sd import load_model, process_features_and_mask, get_mask
from copy import deepcopy

# from sae_editing.train_no_aligner import LightningSAETrainWrapper
from sae_editing.model import HetSAE

def edit_features_with_sae_og(crosscoder, source_features, target_features):
    # Assume that features are stored in the *same* order in the source and target
    crosscoder.eval()
    with torch.no_grad():
        # pdb.set_trace()
        # ____________________________________ #
        # source_key = 'openclip&transformer_resblocks_9_attn_key' 
        # target_key = 'dinov1&encoder_layer_9_attention_attention_key'
        # pdb.set_trace()
        # [B,1,T,D] -> 
        source_features_reshaped = {
            k: reshape_features(v, k).cuda() for k,v in source_features.items()
        }
        target_features_reshaped = {
            k: reshape_features(v, k).cuda() for k,v in target_features.items()
        }
        # source_features[source_key] = reshape_features(source_features[source_key].cpu(), 
        # [B,T,H,E] -> [B,H,T,E] -> [BxHxT,E]
        # target_features[target_key] = target_features[target_key].cpu().reshape(1, 2809, 12, 192).permute(0, 2, 1, 3).flatten(0, 2)
        # pdb.set_trace()
        # ____________________________________ #
        source_encoded = crosscoder.encode(source_features_reshaped, apply_activation=False)
        source_latents = {k: F.relu(v[0]) for k,v in source_encoded.items()}
        target_encoded = crosscoder.encode(target_features_reshaped, apply_activation=False)
        target_latents = {k: F.relu(v[0]) for k,v in target_encoded.items()}
        # # pdb.set_trace()
        # aligned_source = torch.cat([x[0] for x in aligned_source.values()], dim=0)
        # aligned_target = torch.cat([x[0] for x in aligned_target.values()], dim=0)
        # pdb.set_trace()
        # source_latents, source_stats = autoencoder.encode(aligned_source)
        # target_latents, _ = autoencoder.encode(aligned_target)
        # pdb.set_trace()
        fused_latents = {
            source_k: (
                F.relu(target_latents[target_k] - source_latents[source_k]) + source_latents[source_k],
                source_encoded[source_k][1]
            ) for source_k, target_k in zip(source_latents.keys(), target_latents.keys())
        }
        # fused_latents = F.relu(target_latents - source_latents) + source_latents
        # fused_aligned = autoencoder.decode(fused_latents, source_stats)
        # source_key = list(source_features.keys())[0]
        # fused_features = aligner.decode({source_key: fused_aligned})[source_key]
        # pdb.set_trace()
        fused_features = crosscoder.decode(fused_latents)
        fused_features = {
            k: v.reshape(*source_features[k].shape) for k,v in fused_features.items()
        } # Bx1xtx(dxh)
        # pdb.set_trace()
    return get_first_dict_value(fused_features)


def reshape_features(features, key):
    B, _, T, C = features.shape
    try:
        assert _ == 1
    except:
        pdb.set_trace()
    
    if any([tag in key for tag in ['key', 'query', 'value']]):
        features = features.reshape(B, T, 12, C // 12)
    # .permute(0, 2, 1, 3).flatten(start_dim=-2, end_dim=-1)
    return features.flatten(0, 2)


def get_first_dict_value(d):
    return d[list(d.keys())[0]]


def edit_features_with_sae(crosscoder, source_features, target_features):
    # Assume that features are stored in the *same* order in the source and target
    crosscoder.eval()
    with torch.no_grad():
        
        source_features_reshaped = {k: reshape_features(v, k).cuda() for k,v in source_features.items()}
        source_encoded = crosscoder.encode(source_features_reshaped, apply_activation=False)
        source_latents = {k: F.relu(v[0]) for k,v in source_encoded.items()}
        
        target_features_reshaped = {k: reshape_features(v, k).cuda() for k,v in target_features.items()}
        target_encoded = crosscoder.encode(target_features_reshaped, apply_activation=False)
        target_latents = {k: F.relu(v[0]) for k,v in target_encoded.items()}
        
        fused_latents = {
            source_k: (
                F.relu(target_latents[target_k] - source_latents[source_k]) + source_latents[source_k],
                source_encoded[source_k][1]
            ) for source_k, target_k in zip(source_latents.keys(), target_latents.keys())
        }
        fused_features = crosscoder.decode(fused_latents)
        fused_features = {
            k: v.reshape(*source_features[k].shape) for k,v in fused_features.items()
        } # Bx1xtx(dxh)
    return get_first_dict_value(fused_features)


def sum_features(source_features, target_features):
    # Assume that features are stored in the *same* order in the source and target
    with torch.no_grad():
        source_features = {k: v.cuda() for k,v in source_features.items()}
        target_features = {k: v.cuda() for k,v in target_features.items()}
        fused_features = {
            source_k: (
                (source_features[source_k] + target_features[target_k]) / 2
            ) for source_k, target_k in zip(source_features.keys(), target_features.keys())
        }
    return get_first_dict_value(fused_features)


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    if not COUNT_INVIS:
        kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale

def load_spair_data(path, size=256, category='cat', split='test', subsample=None):
    np.random.seed(SEED)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    logger.info(f'Number of SPairs for {category} = {len(pairs)}')
    files = []
    thresholds = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    logger.info(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        
        thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]))

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, thresholds

def load_pascal_data(path, size=256, category='cat', split='test', subsample=None):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(SEED)
    files = []
    kps = []
    test_data = pd.read_csv(f'{path}/{split}_pairs_pf_pascal.csv')
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    logger.info(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    point_A_coords = subset_pairs.iloc[:,3:5]
    point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1,0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size
        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None

def compute_pck(
    sae_type,
    model, aug, save_path, files, kps, category, 
    mask=False, dist='cos', thresholds=None, 
    real_size=960, layer=9, facet='key'
):
    import sys
    sys.path.append('/gscratch/krishna/gstoica3/research/sd-dino/sae_editing')
    # pdb.set_trace()
    # sae_dir = '/mmfs1/gscratch/krishna/gstoica3/research/sae_editing/notebooks/checkpoints'
    sae_dir = '/mmfs1/gscratch/krishna/gstoica3/research/sae_editing/notebooks/explorations/full_sae/checkpoints/vanilla_sae'

    # aligner = torch.load(os.path.join(sae_dir, 'aligner_epoch=9-step=80810_torch_model.pth'))
    # autoencoder = torch.load(os.path.join(sae_dir, 'autoencoder_epoch=9-step=80810_torch_model.pth'))
    # crosscoder = torch.load(os.path.join(sae_dir, 'sae_epoch=4-val_total_loss=0.8.ckpt')).cuda()
    # crosscoder = torch.load(os.path.join(sae_dir, 'epoch=19-val_loss=0.76.ckpt')).cuda()
    # crosscoder = torch.load(os.path.join(sae_dir, 'epoch=29-val_loss=0.62.ckpt')).cuda()
    crosscoder = torch.load(os.path.join(sae_dir, 'epoch=168-val_loss=0.64_orig_mod.ckpt')).cuda()
    # pdb.set_trace()
    img_size = 224
    model_dict={
        'hf_dinov1_vitb16': 'facebook/dino-vitb16',
        'openclip_vitb16': ( 'ViT-B-16', 'laion400m_e31')
    }
    # model_types = [model_dict[model_name] for model_name in MODEL_NAMES]
    # TODO: DELETE THESE
    model_types = [
        ( 'ViT-B-16', 'laion400m_e31'),
        'facebook/dino-vitb16',
    ]
    MODEL_NAMES = [
        'openclip_vitb16',
        'hf_dinov1_vitb16',
    ]
    stride = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    name2processing = {}
    for model_name, model_type in zip(MODEL_NAMES, model_types):
        if 'hf_dinov1_vitb16' == model_name:
            model_name = 'dinov1'
            logger.info("Using HF DINOv1 model")
            extractor = ViTExtractorHF(model_type, stride, device=device)
            patch_size = extractor.model.embeddings.patch_embeddings.patch_size[0]
        elif 'openclip_vitb16' == model_name:
            model_name = 'openclip'
            logger.info("Using OpenClip model")
            extractor = OpenClipExtractorHF(model_type, stride, device=device)
            patch_size = extractor.model.patch_size[0]
        else:
            extractor = ViTExtractor(model_type, stride, device=device)
            patch_size = extractor.model.embeddings.patch_size
        num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
        name2processing[model_name] = {
            'extractor': extractor,
            'patch_size': patch_size,
            'num_patches': num_patches
        }

    current_save_results = 0
    gt_correspondences = []
    pred_correspondences = []
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    pbar = tqdm(total=N)

    for pair_idx in range(N):
        # pdb.set_trace()
        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1_kps = kps[2*pair_idx]

        # Get patch index for the keypoints
        img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()
        img1_y_patch = (num_patches / img_size * img1_y).astype(np.int32)
        img1_x_patch = (num_patches / img_size * img1_x).astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch

        # Load image 2
        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2_kps = kps[2*pair_idx+1]

        # Get patch index for the keypoints
        img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()
        img2_y_patch = (num_patches / img_size * img2_y).astype(np.int32)
        img2_x_patch = (num_patches / img_size * img2_x).astype(np.int32)
        img2_patch_idx = num_patches * img2_y_patch + img2_x_patch

        # pdb.set_trace()
        dist = dist.lower()
        def extract_features(extractor, layer, facet, dist, name):
            with torch.no_grad():
                if not CO_PCA:
                    # if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

                else: # THIS IS HIT
                    # if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)
                
                if CO_PCA_DINO: # THIS IS IGNORED
                    cat_desc_dino = torch.cat((img1_desc_dino, img2_desc_dino), dim=2).squeeze() # (1, 1, num_patches**2, dim)
                    mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
                    centered_features = cat_desc_dino - mean
                    U, S, V = torch.pca_lowrank(centered_features, q=CO_PCA_DINO)
                    reduced_features = torch.matmul(centered_features, V[:, :CO_PCA_DINO]) # (t_x+t_y)x(d)
                    processed_co_features = reduced_features.unsqueeze(0).unsqueeze(0)
                    img1_desc_dino = processed_co_features[:, :, :img1_desc_dino.shape[2], :]
                    img2_desc_dino = processed_co_features[:, :, img1_desc_dino.shape[2]:, :]

                if 'l1' in dist or 'l2' in dist or dist == 'plus_norm': # THIS IS IGNORED
                    # normalize the features
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

                if dist=='plus' or dist=='plus_norm': # THIS IS IGNORED
                    img1_desc = img1_desc + img1_desc_dino
                    img2_desc = img2_desc + img2_desc_dino
                    dist='cos'
                
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino
            module_name = extractor.hook_module_name.replace('.', '_')
            return {
                'img1_desc': {f'{name}&{module_name}': img1_desc.cpu()}, 
                'img2_desc': {f'{name}&{module_name}': img2_desc.cpu()}
            }
        
        img1_descs = []
        img2_descs = []
        for name, processing in name2processing.items():
            descs = extract_features(processing['extractor'], layer, facet, dist, name)
            img1_descs.append(descs['img1_desc'])
            img2_descs.append(descs['img2_desc'])
        
        # pdb.set_trace()
        if sae_type == 'dino_to_clip':
            # print("dino_to_clip")
            img1_desc = edit_features_with_sae(crosscoder, img1_descs[0], img1_descs[1]).cuda()
            img2_desc = edit_features_with_sae(crosscoder, img2_descs[0], img2_descs[1]).cuda()
        elif sae_type == 'clip_to_dino':
            # print("clip_to_dino")
            img1_desc = edit_features_with_sae(crosscoder, img1_descs[1], img1_descs[0]).cuda()
            img2_desc = edit_features_with_sae(crosscoder, img2_descs[1], img2_descs[0]).cuda()
        elif sae_type == 'ensemble_sum':
            # print("ensemble_sum")
            img1_desc = sum_features(img1_descs[0], img1_descs[1]).cuda()
            img2_desc = sum_features(img2_descs[0], img2_descs[1]).cuda()
        else:
            raise ValueError("Uknown SAE type")
        # pdb.set_trace()
        # logger.info(img1_desc.shape, img2_desc.shape)
        if MASK and CO_PCA: # THIS IS IGNORED
            mask2 = get_mask(model, aug, img2, category)
            img2_desc = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
            resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest')
            img2_desc = img2_desc * resized_mask2.repeat(1, img2_desc.shape[1], 1, 1)
            img2_desc[(img2_desc.sum(dim=1)==0).repeat(1, img2_desc.shape[1], 1, 1)] = 100000
            # reshape back
            img2_desc = img2_desc.reshape(1, 1, img2_desc.shape[1], num_patches*num_patches).permute(0,1,3,2)

        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        if COUNT_INVIS:
            vis = torch.ones_like(vis)
        # Get similarity matrix
        # pdb.set_trace()
        if dist == 'cos':
            sim_1_to_2 = chunk_cosine_sim(img1_desc, img2_desc).squeeze()
        elif dist == 'l2':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2).squeeze()
        elif dist == 'l1':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1).squeeze()
        elif dist == 'l2_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2, normalize=True).squeeze()
        elif dist == 'l1_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1, normalize=True).squeeze()
        else:
            raise ValueError('Unknown distance metric')

        # Get nearest neighors
        nn_1_to_2 = torch.argmax(sim_1_to_2[img1_patch_idx], dim=1)
        nn_y_patch, nn_x_patch = nn_1_to_2 // num_patches, nn_1_to_2 % num_patches
        nn_x = (nn_x_patch - 1) * stride + stride + patch_size // 2 - .5
        nn_y = (nn_y_patch - 1) * stride + stride + patch_size // 2 - .5
        kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)

        gt_correspondences.append(img2_kps[vis][:, [1,0]])
        pred_correspondences.append(kps_1_to_2[vis][:, [1,0]])
        if thresholds is not None:
            bbox_size.append(thresholds[pair_idx].repeat(vis.sum()))
        
        if current_save_results!=TOTAL_SAVE_RESULT:
            tmp_alpha = torch.tensor([0.1, 0.05, 0.01])
            if thresholds is not None:
                tmp_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                tmp_threshold = tmp_alpha.unsqueeze(-1) * tmp_bbox_size.unsqueeze(0)
            else:
                tmp_threshold = tmp_alpha * img_size
            if not os.path.exists(f'{save_path}/{category}'):
                os.makedirs(f'{save_path}/{category}')
            # fig=draw_correspondences_gathered(img1_kps[vis][:, [1,0]], kps_1_to_2[vis][:, [1,0]], img1, img2)
            fig=draw_correspondences_lines(img1_kps[vis][:, [1,0]], kps_1_to_2[vis][:, [1,0]], img2_kps[vis][:, [1,0]], img1, img2, tmp_threshold)
            fig.savefig(f'{save_path}/{category}/{pair_idx}_pred.png')
            fig_gt=draw_correspondences_gathered(img1_kps[vis][:, [1,0]], img2_kps[vis][:, [1,0]], img1, img2)
            fig_gt.savefig(f'{save_path}/{category}/{pair_idx}_gt.png')
            plt.close(fig)
            plt.close(fig_gt)
            current_save_results+=1

        pbar.update(1)
    # pdb.set_trace()
    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01]) if not PASCAL else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))

    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct = err < threshold
    else:
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)

    correct = correct.sum(dim=-1) / len(gt_correspondences)

    alpha2pck = zip(alpha.tolist(), correct.tolist())
    logger.info(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                    for alpha, pck_alpha in alpha2pck]))

    return correct

def main(args):
    global SAE_TYPE, MASK, SAMPLE, DIST, COUNT_INVIS, TOTAL_SAVE_RESULT, BBOX_THRE, VER, CO_PCA, PCA_DIMS, SIZE, FUSE_DINO, DINOV2, MODEL_NAMES, TEXT_INPUT, ONLY_DINO, SEED, EDGE_PAD, WEIGHT, CO_PCA_DINO, PASCAL, RAW
    MASK = args.MASK
    SAMPLE = args.SAMPLE
    DIST = args.DIST
    COUNT_INVIS = args.COUNT_INVIS
    TOTAL_SAVE_RESULT = args.TOTAL_SAVE_RESULT
    BBOX_THRE = False if args.IMG_THRESHOLD else True
    VER = args.VER
    CO_PCA = False if args.PROJ_LAYER else True
    CO_PCA_DINO = args.CO_PCA_DINO
    PCA_DIMS = args.PCA_DIMS
    SIZE = args.SIZE
    INDICES = args.INDICES
    EDGE_PAD = args.EDGE_PAD

    FUSE_DINO = True
    ONLY_DINO = args.ONLY_DINO
    DINOV2 = False if args.DINOV1 else True
    MODEL_NAMES = args.MODEL_NAMES
    TEXT_INPUT = args.TEXT_INPUT
    
    SEED = args.SEED
    WEIGHT = args.WEIGHT # corresponde to three groups for the sd features, and one group for the dino features
    PASCAL = args.PASCAL
    RAW = args.RAW
    SAE_TYPE = args.SAE_TYPE
    
    SAE_CHECKPOINT_PATH = args.SAE_CHECKPOINT_PATH
    
    LAYERS = [9]
    # LAYERS = [9, 10, 11]
    # FACETS = ['key', 'query', 'value', 'token'][::-1]
    FACETS = ['key']
    
    if SAMPLE == 0:
        SAMPLE = None
    if ONLY_DINO:
        FUSE_DINO = True
    if FUSE_DINO and not ONLY_DINO:
        DIST = "l2"
    else:
        DIST = "cos"
    if args.DIST != "cos" and args.DIST != "l2":
        DIST = args.DIST
    if PASCAL:
        SAMPLE = 0

    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    model, aug = load_model(
        diffusion_ver=VER, image_size=SIZE, 
        num_timesteps=args.TIMESTEP, 
        block_indices=tuple(INDICES)
    )
    save_path=f'./results_spair/pck_fuse_{args.NOTE}mask_{MASK}_sample_{SAMPLE}_BBOX_{BBOX_THRE}_dist_{DIST}_Invis_{COUNT_INVIS}_{args.TIMESTEP}{VER}_{MODEL_NAMES}_{SIZE}_copca_{CO_PCA}_{INDICES[0]}_{PCA_DIMS[0]}_{INDICES[1]}_{PCA_DIMS[1]}_{INDICES[2]}_{PCA_DIMS[2]}_text_{TEXT_INPUT}_sd_{WEIGHT[3]}{not ONLY_DINO}_dino_{WEIGHT[4]}{FUSE_DINO}'
    if PASCAL:
        save_path=f'./results_pascal/pck_fuse_{args.NOTE}mask_{MASK}_sample_{SAMPLE}_BBOX_{BBOX_THRE}_dist_{DIST}_Invis_{COUNT_INVIS}_{args.TIMESTEP}{VER}_{MODEL_NAMES}_{SIZE}_copca_{CO_PCA}_{INDICES[0]}_{PCA_DIMS[0]}_{INDICES[1]}_{PCA_DIMS[1]}_{INDICES[2]}_{PCA_DIMS[2]}_text_{TEXT_INPUT}_sd_{WEIGHT[3]}{not ONLY_DINO}_dino_{WEIGHT[4]}{FUSE_DINO}'
    if EDGE_PAD:
        save_path += '_edge_pad'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger = get_logger(save_path+'/result.log')

    logger.info(args)
    data_dir = '/gscratch/krishna/gstoica3/datasets/spair71k/SPair-71k/' if not PASCAL else 'data/PF-dataset-PASCAL'
    if not PASCAL:
        categories = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
        categories = sorted(categories)
    else:
        categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # for pascal
    # img_size = 840 if DINOV2 else 224 if ONLY_DINO else 480
    img_size = 224
    best_pcks = None
    start_time=time.time()
    for layer in LAYERS:
        for facet in FACETS:
            logger.info("-"*50)
            logger.info(f"Layer: {layer}, Facet: {facet}")
            config_pcks = {
                'categories': categories + ['average'],
                'PCK0.10': [],
                'PCK0.05': [],
                'PCK0.01': []
            }
            pcks = []
            pcks_05 = []
            pcks_01 = []
            for cat in categories:
                # pdb.set_trace()
                # files, kps, thresholds = load_spair_data(data_dir, size=img_size, category=cat, subsample=SAMPLE) if not PASCAL else load_pascal_data(data_dir, size=img_size, category=cat, subsample=SAMPLE)
                if PASCAL:
                    files, kps, thresholds = load_pascal_data(data_dir, size=img_size, category=cat, subsample=SAMPLE)
                else:
                    files, kps, thresholds = load_spair_data(data_dir, size=img_size, category=cat, subsample=SAMPLE)
                    
                if BBOX_THRE:
                    pck = compute_pck(
                        SAE_TYPE,
                        model, aug, save_path, files, kps, cat, mask=MASK, 
                        dist=DIST, thresholds=thresholds, real_size=SIZE,
                        layer=layer, facet=facet
                    )
                else:
                    pck = compute_pck(
                        SAE_TYPE,
                        model, aug, save_path, files, kps, cat,
                        mask=MASK, dist=DIST, real_size=SIZE,
                        layer=layer, facet=facet
                    )
                pcks.append(pck[0])
                pcks_05.append(pck[1])
                pcks_01.append(pck[2])
                
                config_pcks['PCK0.10'] += [pck[0]]
                config_pcks['PCK0.05'] += [pck[1]]
                config_pcks['PCK0.01'] += [pck[2]]
                
            config_pcks['PCK0.10'] += [np.average(pcks)]
            config_pcks['PCK0.05'] += [np.average(pcks_05)]
            config_pcks['PCK0.01'] += [np.average(pcks_01)]
            
            end_time=time.time()
            minutes, seconds = divmod(end_time-start_time, 60)
            logger.info(f"Time: {minutes:.0f}m {seconds:.0f}s")
            logger.info(f"Average PCK0.10: {np.average(pcks) * 100:.2f}")
            logger.info(f"Average PCK0.05: {np.average(pcks_05) * 100:.2f}")
            logger.info(f"Average PCK0.01: {np.average(pcks_01) * 100:.2f}") if not PASCAL else logger.info(f"Average PCK0.15: {np.average(pcks_01) * 100:.2f}")

            # pdb.set_trace()
            if best_pcks is None or best_pcks['PCK0.10'][-1] < config_pcks['PCK0.10'][-1]:
                logger.info(f"New best PCK0.10: {config_pcks['PCK0.10'][-1] * 100:.2f}")
                best_pcks = deepcopy(config_pcks)
                best_pcks['layer'] = layer
                best_pcks['facet'] = facet
    
    for idx in range(len(best_pcks['categories'])):
        logger.info(f"{best_pcks['categories'][idx]}:")
        logger.info(f"PCK0.10: {best_pcks['PCK0.10'][idx] * 100:.2f}")
        logger.info(f"PCK0.05: {best_pcks['PCK0.05'][idx] * 100:.2f}")
        logger.info(f"PCK0.01: {best_pcks['PCK0.01'][idx] * 100:.2f}")
    logger.info(f"Best layer: {best_pcks['layer']}")
    logger.info(f"Best facet: {best_pcks['facet']}")
    
    pcks = best_pcks['PCK0.10'][:-1]
    pcks_05 = best_pcks['PCK0.05'][:-1]
    pcks_01 = best_pcks['PCK0.01'][:-1]
    
    if SAMPLE is None or SAMPLE==0:
        weights_pascal=[15,30,10,6,8,32,19,27,13,3,8,24,9,27,12,7,1,13,20,15]
        weights_spair=[690,650,702,702,870,644,564,600,646,640,600,600,702,650,862,664,756,692]
        weights = weights_pascal if PASCAL else weights_spair
    else:
        weights = [1] * len(pcks)
    logger.info(f"Weighted PCK0.10: {np.average(pcks, weights=weights) * 100:.2f}")
    logger.info(f"Weighted PCK0.05: {np.average(pcks_05, weights=weights) * 100:.2f}")
    logger.info(f"Weighted PCK0.01: {np.average(pcks_01, weights=weights) * 100:.2f}") if not PASCAL else logger.info(f"Weighted PCK0.15: {np.average(pcks_01, weights=weights) * 100:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--MASK', action='store_true', default=False)               # set true to use the segmentation mask for the extracted features
    parser.add_argument('--SAMPLE', type=int, default=20)                           # sample 20 pairs for each category, set to 0 to use all pairs
    parser.add_argument('--DIST', type=str, default='l2')                           # distance metric, cos, l2, l1, l2_norm, l1_norm, plus, plus_norm
    parser.add_argument('--COUNT_INVIS', action='store_true', default=False)        # set true to count invisible keypoints
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=5)                 # save the qualitative results for the first 5 pairs
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set the pck threshold to the image size rather than the bbox size
    parser.add_argument('--VER', type=str, default="v1-5")                          # version of diffusion, v1-3, v1-4, v1-5, v2-1-base
    parser.add_argument('--PROJ_LAYER', action='store_true', default=False)         # set true to use the pretrained projection layer from ODISE for dimension reduction
    parser.add_argument('--CO_PCA_DINO', type=int, default=0)                       # whether perform co-pca on dino features
    parser.add_argument('--PCA_DIMS', nargs=3, type=int, default=[256, 256, 256])   # the dimensions of the three groups of sd features
    parser.add_argument('--TIMESTEP', type=int, default=100)                        # timestep for diffusion, [0, 1000], 0 for no noise added
    parser.add_argument('--SIZE', type=int, default=960)                            # image size for the sd input
    parser.add_argument('--INDICES', nargs=4, type=int, default=[2,5,8,11])         # select different layers of sd features, only the first three are used by default
    parser.add_argument('--EDGE_PAD', action='store_true', default=False)           # set true to pad the image with the edge pixels
    parser.add_argument('--WEIGHT', nargs=5, type=float, default=[1,1,1,1,1])       # first three corresponde to three layers for the sd features, and the last two for the ensembled sd/dino features
    parser.add_argument('--RAW', action='store_true', default=False)                # set true to use the raw features from sd

    parser.add_argument('--NOT_FUSE', action='store_true', default=False)           # set true to use only sd features
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)          # set true to use only dino features
    parser.add_argument('--DINOV1',  action='store_true', default=False)            # set true to use dinov1
    parser.add_argument('--MODEL_NAMES', type=str, default='base')                   # model size of thye dinov2, small, base, large
    parser.add_argument('--TEXT_INPUT', action='store_true', default=False)         # set true to use the explicit text input

    parser.add_argument('--PASCAL', action='store_true', default=False)             # set true to test on pfpascal dataset
    parser.add_argument('--NOTE', type=str, default='')
    parser.add_argument('--SAE_CHECKPOINT_PATH', type=str, default='/mmfs1/gscratch/krishna/gstoica3/research/sae_editing/notebooks/checkpoints/epoch=9-step=80810_torch.pth')
    parser.add_argument('--SAE_TYPE', type=str, default='dino_to_clip')
    args = parser.parse_args()
    main(args)
