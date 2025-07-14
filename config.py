"""
This config is so it's easy to find which birds classify. It also allows for choosing which classifiers to train.
While I can run from the command line, I built this in an IDE, and find it easier to just change variables here.
"""


species_to_scrape = ['Blue-footed Booby', 'Western Barn Owl', 'Varied Thrush', 'American Bittern', 'Eastern Cattle Egret',
                     'Black-capped Chickadee', 'Gyrfalcon', 'Bald Eagle', 'Atlantic Puffin', 'Common Loon', 'Sandhill Crane',
                     'Varied Thrush', 'Cedar Waxwing', 'Great Kiskadee']

data_dir = "Data"

SPECTROGRAM_SHAPE = (128, 432)

# Training config
TRAIN_CNN = True
TRAIN_TRANSFORMER = False
USE_GRID_SEARCH = True

# Training parameters
CNN_EPOCHS = 20
TRANSFORMER_EPOCHS = 10
TRANSFORMER_BATCH_SIZE = 8
TRANSFORMER_MODEL = "facebook/wav2vec2-base"

# Grid search parameters
CNN_GRID = {
    "epochs": [15, 20, 25, 30]
}

TRANSFORMER_GRID = {
    "epochs": [5, 10, 15],
    "batch_size": [4, 8, 16],
    "model_name": ["facebook/wav2vec2-base"]
}