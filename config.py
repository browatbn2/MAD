import os
import math


PROJECT_DIR = '/home/browatbn/dev/expressions'
DATA_DIR = os.path.join('/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/dev', 'data')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

MODEL_DIR = os.path.join(DATA_DIR, 'models')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
SNAPSHOT_DIR = os.path.join(MODEL_DIR, 'snapshots')

AFFECTNET_RAW_DIR = '/home/browatbn/dev/datasets/AffectNet'
AFFECTNET_ROOT = '/home/browatbn/dev/datasets/AffectNet'

VGGFACE2_ROOT = '/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/datasets/VGGFace2'
VGGFACE2_ROOT_LOCAL = '/home/browatbn/dev/datasets/VGGFace2'

VOXCELEB_ROOT = '/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/datasets/VoxCeleb1'
VOXCELEB_ROOT_LOCAL = '/home/browatbn/dev/datasets/VoxCeleb1'

LFW_ROOT = '/home/browatbn/dev/datasets/LFW'

CELEBA_ROOT = '/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/datasets/CelebA'
CELEBA_ROOT_LOCAL = '/home/browatbn/dev/datasets/CelebA'

ARCH = 'resnet'

INPUT_SCALE_FACTOR = 2
INPUT_SIZE = 128 * INPUT_SCALE_FACTOR
# CROP_BORDER = 6 * INPUT_SCALE_FACTOR
# CROP_SIZE = INPUT_SIZE + CROP_BORDER * 2
CROP_SIZE = math.ceil(INPUT_SIZE * 2**0.5)  # crop size equals input diagonal, so images can be fully rotated
CROP_BORDER = CROP_SIZE - INPUT_SIZE

CROP_BY_EYE_MOUTH_DIST = False
CROP_ALIGN_ROTATION = False
CROP_SQUARE = True

# crop resizing params for cropping based on landmarks
CROP_MOVE_TOP_FACTOR = 0.2       # move top by factor of face heigth in respect to eye brows
CROP_MOVE_BOTTOM_FACTOR = 0.12   # move bottom by factor of face heigth in respect to chin bottom point

MIN_OPENFACE_CONFIDENCE = 0.4

ENCODER_LAYER_NORMALIZATION = 'batch'
DECODER_LAYER_NORMALIZATION = 'batch'
ENCODING_DISTRIBUTION = 'normal'

DECODER_FIXED_ARCH = True
DECODER_PLANES_PER_BLOCK = 1

# Autoencoder losses
TRAIN_ENCODER = True
TRAIN_DECODER = True

RGAN = False
UPDATE_DISCRIMINATOR_FREQ = 4
UPDATE_ENCODER_FREQ = 1

WITH_ZGAN = True
WITH_GAN = True
WITH_PERCEPTUAL_LOSS = False
WITH_BUMP_LOSS = False
WITH_CYCLE_LOSS = False
WITH_GEN_LOSS = True
WITH_LANDMARK_LOSS = False
WITH_SSIM_LOSS = True

WITH_HIST_NORM = False
WITH_PARALLEL_DISENTANGLEMENT = True

# Recontruction loss
W_RECON = 1.0
W_SSIM = 60.0

NFT = 16
EXPRESSION_DIMS = 32

# Disentanglement losses
WITH_DISENT_CYCLE_LOSS = True
WITH_AUGMENTATION_LOSS = True
WITH_FEATURE_LOSS = True
WITH_Z_RECON_LOSS = True

WITH_POSE = False

# Disentanglement weights
# W_Z_RECON = 1.0
W_Z_RECON = 4.0
W_FEAT = 1.0
W_CYCLE = 1.0
W_AUG = 1.0

W_DISENT = 5.0

HARD_TRIPLETS_FOR_IDENTITY = True

WITH_FACE_MASK = False
WITH_RECON_ERROR_WEIGHTING = False

EDIT_FACES = False
SHOW_TRIPLETS = False
SHOW_RANDOM_FACES = True

WEIGHT_RECON_LOSS = 2.0

WITH_FOURIER_TRANSFORM = False

LOOSE_BBOX_SCALE = 1.4

CURRENT_MODEL = os.path.join(SNAPSHOT_DIR, 'exp_vgg_disent_joint/00072')

