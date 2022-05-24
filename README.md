# FMST-PT
Implementation of paper "FMSF-PT: A fast 3D medical image segmentation framework guided by phased tasks" using pytorch。

![image](https://user-images.githubusercontent.com/33023091/169638563-d4b4d644-6e2c-44d4-a48c-21d7b67d77e0.png)

The reasoning speed of traditional networks such as swinnet, unetr, nnunet and nnformer is FPS < 1, but our reasoning speed is super fast. You can get any 3D medical impact data after analysis through FPS > 15 processing speed. Tested on FPS 2017_ consenuse>30, FPS_ attention>20, FPS_ edge>15.

And we propose a dice evaluation for compressed data mask, attndice and a corresponding loss function attnloss. When the mask data is 0 or 1, attndice = dice. When the value is between 0 and 1, attndice > dice, which can more accurately judge the edge accuracy of compressed mask.

We uploaded the reasoning results on lits2017 for reference. Location: [All stage](https://github.com/Human9000/FMST-PT/tree/main/val), [Consensus](https://github.com/Human9000/FMST-PT/tree/main/val/attention), [Attention](https://github.com/Human9000/FMST-PT/tree/main/val/consensus), [Edge](https://github.com/Human9000/FMST-PT/tree/main/val/edge), and some of them are shown in the figure below:

- Consensus by `sunet`

![image](https://user-images.githubusercontent.com/33023091/170104382-a47039cb-2737-49c2-993d-7bf6cb5eb5ee.png)

- Attention by `attnnet`

![image](https://user-images.githubusercontent.com/33023091/170104534-09d0311a-42af-4955-8284-0614ee38f20c.png)

- Edge by `edgenet`

![image](https://user-images.githubusercontent.com/33023091/170104617-ccaeb89a-30ae-425e-93ab-ded79bf1f0ed.png)



We disclosed some codes, including network structure in `./model/fmst.py`, It contains three networks: "snet", "attnett" and "edgenet", which correspond to the three stages in the framework.

You can use our program to carry out secondary training one by one through the following commands(You need to modify the optimal weight position specified in the subsequent code after each training!):
```cmd
python train_attention.py
python train_consensus.py
python train_edge.py
```

Data set spectrum and stage attention information are implemented in `./dataset/lits2017.py` as an example. 

In addition, we provided the configuration file format of the data set in `./dataset/lits2017.json` as an example.
```json
{
    "name": "lits2017",
    "labels": {
        "0": "background",
        "1": "liver",
        "2": "tumer"
    },
    "aim_label": "1",
    "modality": {
        "0": "CT"
    },
    "numTest": 71,
    "numTrain": 101,
    "numValid": 30,
    "tensorImageSize": "3D",
    "test": [
        "./Test_Data/test-volume-0.nii",
        "./Test_Data/test-volume-69.nii"
    ],
    "train": [
        {
            "image": "./TrainingData/volume-0.nii",
            "label": "./TrainingData/segmentation-0.nii"
        },
        {
            "image": "./TrainingData/volume-100.nii",
            "label": "./TrainingData/segmentation-100.nii"
        }
    ],
    "valid": [
        {
            "image": "./TrainingData/volume-101.nii",
            "label": "./TrainingData/segmentation-101.nii"
        },
        {
            "image": "./TrainingData/volume-130.nii",
            "label": "./TrainingData/segmentation-130.nii"
        }
    ]
}
```

