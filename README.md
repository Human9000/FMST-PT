# FMST-PT
FMSF-PT: A fast 3D medical image segmentation framework guided by phased tasks

![image](https://user-images.githubusercontent.com/33023091/169638563-d4b4d644-6e2c-44d4-a48c-21d7b67d77e0.png)


We disclosed some codes, including network structure in `./model/fmst.py`, It contains three networks: "snet", "attnett" and "edgenet", which correspond to the three stages in the framework

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
- Consensus by `sunet`

![image](https://user-images.githubusercontent.com/33023091/170104382-a47039cb-2737-49c2-993d-7bf6cb5eb5ee.png)

- Attention by `attnnet`

![image](https://user-images.githubusercontent.com/33023091/170104534-09d0311a-42af-4955-8284-0614ee38f20c.png)

- Edge by `edgenet`

![image](https://user-images.githubusercontent.com/33023091/170104617-ccaeb89a-30ae-425e-93ab-ded79bf1f0ed.png)



