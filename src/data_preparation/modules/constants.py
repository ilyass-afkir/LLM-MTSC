"""
Constants and mappings used for data preparation, labeling, encoding, and dataset configuration.

Contents:
- UEA dataset configuration parameters
"""

UEA = {
    "ArticularyWordRecognition": {
        "num_channels": 9, "num_classes": 25, "sequence_length": 144,
        "patch_length": 20, "patch_stride": 10, "batch_size": 64
    },
    "AtrialFibrillation": {
        "num_channels": 2, "num_classes": 3, "sequence_length": 640,
        "patch_length": 50, "patch_stride": 25, "batch_size": 5
    },
    "BasicMotions": {
        "train_size": 40, "test_size": 40,
        "num_channels": 6, "sequence_length": 100, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 10
    },
    "CharacterTrajectories": {
        "num_channels": 3, "sequence_length": 182, "num_classes": 20,
        "patch_length": 128, "patch_stride": 64, "batch_size": 256
    },
    "Cricket": {
        "num_channels": 6, "sequence_length": 1197, "num_classes": 12,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "DuckDuckGeese": {
        "train_size": 60, "test_size": 40,
        "num_channels": 1345, "sequence_length": 270, "num_classes": 5,
        "patch_length": 32, "patch_stride": 16, "batch_size": 10
    },
    "EigenWorms": {
        "num_channels": 6, "num_classes": 5, "sequence_length": 17984,
        "patch_length": 256, "patch_stride": 128, "batch_size": 2
    },
    "Epilepsy": {
        "num_channels": 3, "sequence_length": 206, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "EthanolConcentration": {
        "num_channels": 3, "sequence_length": 1751, "num_classes": 4,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "ERing": {
        "num_channels": 4, "sequence_length": 65, "num_classes": 6,
        "patch_length": 8, "patch_stride": 4, "batch_size": 8
    },
    "FaceDetection": {
        "num_channels": 144, "sequence_length": 62, "num_classes": 2,
        "patch_length": 8, "patch_stride": 4, "batch_size": 256
    },
    "FingerMovements": {
        "num_channels": 28, "sequence_length": 50, "num_classes": 2,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "HandMovementDirection": {
        "num_channels": 10, "sequence_length": 400, "num_classes": 4,
        "patch_length": 64, "patch_stride": 32, "batch_size": 64
    },
    "Handwriting": {
        "num_channels": 3, "sequence_length": 152, "num_classes": 26,
        "patch_length": 32, "patch_stride": 16, "batch_size": 32
    },
    "Heartbeat": {
        "num_channels": 61, "sequence_length": 405, "num_classes": 2,
        "patch_length": 64, "patch_stride": 32, "batch_size": 32
    },
    "JapaneseVowels": {
        "num_channels": 12, "sequence_length": 29, "num_classes": 9,
        "patch_length": 8, "patch_stride": 4, "batch_size": 64
    },
    "Libras": {
        "num_channels": 2, "sequence_length": 45, "num_classes": 15,
        "patch_length": 8, "patch_stride": 4, "batch_size": 64
    },
    "LSST": {
        "num_channels": 6, "sequence_length": 36, "num_classes": 14,
        "patch_length": 2, "patch_stride": 2, "batch_size": 256
    },
    "InsectWingbeat": {
        "num_channels": 200, "sequence_length": 78, "num_classes": 10,
        "patch_length": 16, "patch_stride": 8, "batch_size": 128
    },
    "MotorImagery": {
        "num_channels": 64, "sequence_length": 3000, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "NATOPS": {
        "num_channels": 24, "sequence_length": 51, "num_classes": 6,
        "patch_length": 16, "patch_stride": 8, "batch_size": 32
    },
    "PenDigits": {
        "num_channels": 2, "sequence_length": 8, "num_classes": 10,
        "patch_length": 2, "patch_stride": 1, "batch_size": 256
    },
    "PEMS-SF": {
        "num_channels": 963, "sequence_length": 144, "num_classes": 7,
        "patch_length": 16, "patch_stride": 8, "batch_size": 32
    },
    "Phoneme": {
        "num_channels": 11, "sequence_length": 217, "num_classes": 39,
        "patch_length": 16, "patch_stride": 8, "batch_size": 256
    },
    "RacketSports": {
        "num_channels": 6, "sequence_length": 30, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "SelfRegulationSCP1": {
        "num_channels": 6, "sequence_length": 896, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "SelfRegulationSCP2": {
        "num_channels": 7, "sequence_length": 1152, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "SpokenArabicDigits": {
        "num_channels": 13, "sequence_length": 93, "num_classes": 10,
        "patch_length": 16, "patch_stride": 8, "batch_size": 256
    },
    "StandWalkJump": {
        "num_channels": 4, "sequence_length": 2500, "num_classes": 3,
        "patch_length": 128, "patch_stride": 64, "batch_size": 1
    },
    "UWaveGestureLibrary": {
        "num_channels": 3, "sequence_length": 315, "num_classes": 8,
        "patch_length": 32, "patch_stride": 16, "batch_size": 32
    }
}