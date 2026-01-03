import torch

print(torch.__version__)

import numpy as np
import dgl

# feat_dict=np.load('../data/ABIDE/modal_feat_dict.npy',allow_pickle=True)
# graph=np.load('../data/ABIDE/ABIDE_weighted-cosine_graph.npz',allow_pickle=True)
# print(feat_dict.item().keys())

# feat=np.load('../magb/Movies/MMFeature/Movies_Llama-3.2-11B-Vision-Instruct_tv.npy',allow_pickle=True)
feat=np.load('../magb/Movies/MMFeature/Movies_LLAMA8B_CLIP.npy',allow_pickle=True)
# feat=np.load('../magb/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy',allow_pickle=True)
# feat=np.load('../magb/Movies/TextFeature/Movies_roberta_base_512_mean.npy',allow_pickle=True)
vfeat=np.load('../magb/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy',allow_pickle=True)
tfeat=np.load('../magb/Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy',allow_pickle=True)

graph=dgl.load_graphs('../magb/Movies/MoviesGraph.pt')[0][0]
# graph=torch.load('../magb/Movies/MoviesGraph.pt')
print((feat==np.hstack((tfeat,vfeat))).all())
