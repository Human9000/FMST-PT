# FMST-PT
Implementation of paper [FMSF-PT: A fast 3D medical image segmentation framework guided by phased tasks](https://openreview.net/pdf?id=q5PStJkall) using pytorch.

![image](https://user-images.githubusercontent.com/33023091/169638563-d4b4d644-6e2c-44d4-a48c-21d7b67d77e0.png)

The reasoning speed of traditional networks such as swinnet, unetr, nnunet and nnformer is FPS < 1, but our reasoning speed is super fast. You can get any 3D medical impact data after analysis through FPS > 15 processing speed. Tested on FPS 2017_ consenuse>30, FPS_ attention>20, FPS_ edge>15.

And we propose a dice evaluation for compressed data mask, attndice and a corresponding loss function attnloss. When the mask data is 0 or 1, attndice = dice. When the value is between 0 and 1, attndice > dice, which can more accurately judge the edge accuracy of compressed mask.



We disclosed some codes, including network structure in `./model/fmst.py`, It contains three networks: "snet", "attnett" and "edgenet", which correspond to the three stages in the framework.

You can use our program to carry out secondary training one by one through the following commands(You need to modify the optimal weight position specified in the subsequent code after each training!):
```cmd
python train_attention.py
python train_consensus.py
python train_edge.py
```

We uploaded the reasoning results on lits2017 for reference. Location: [All stage](https://github.com/Human9000/FMST-PT/tree/main/val), [Consensus](https://github.com/Human9000/FMST-PT/tree/main/val/attention), [Attention](https://github.com/Human9000/FMST-PT/tree/main/val/consensus), [Edge](https://github.com/Human9000/FMST-PT/tree/main/val/edge), and some of them are shown in the figure below:

| `sunet` | `attnnet` | `edgenet` |
| --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/33023091/170104382-a47039cb-2737-49c2-993d-7bf6cb5eb5ee.png) | ![image](https://user-images.githubusercontent.com/33023091/170104534-09d0311a-42af-4955-8284-0614ee38f20c.png) | ![image](https://user-images.githubusercontent.com/33023091/170104617-ccaeb89a-30ae-425e-93ab-ded79bf1f0ed.png) |




We try to train and infer 14 classifications on `btcv synapse` data set based on the weight of `lits 2017` liver 2 classification, which is only 2 hours on rtx3090, and give the output results of the visualization framework. We used 690 CT as the training set and 180 CT as the test set, and the reasoning speed of the model can still be maintained at FPS > 10.

![image](https://user-images.githubusercontent.com/33023091/170417497-a469b662-f8a1-49b7-be04-08645969f77d.png)


