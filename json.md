
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
    "aim_label": ["1","2"],
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
