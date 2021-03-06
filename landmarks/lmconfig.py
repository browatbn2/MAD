LANDMARK_TARGET = 'multi_channel'
# LANDMARK_TARGET = 'hamming'
LANDMARK_SIGMA = 7.0
MIN_LANDMARK_CONF = 0.45

LANDMARK_OCULAR_NORM = 'pupil'
PREDICT_HEATMAP = True

ALL_LANDMARKS = list(range(0, 68))
LANDMARKS_6 = [36, 39, 42, 45, 48, 54]
LANDMARKS_9 = [30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_12 = [21, 22, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_19 = [0, 4, 8, 12, 16, 17, 21, 22, 26, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_22 = [0, 4, 8, 12, 16, 17, 21, 22, 26, 27, 28, 29, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_14 = [17, 26, 21, 22, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS = ALL_LANDMARKS
LANDMARKS_NO_OUTLINE = list(range(17,68))
LANDMARKS_ONLY_OUTLINE = list(range(17))
# LANDMARKS_TO_EVALUATE = LANDMARKS_6
# LANDMARKS_TO_EVALUATE = LANDMARKS_ONLY_OUTLINE
# LANDMARKS_TO_EVALUATE = [0, 4, 8, 12, 16]
# LANDMARKS_TO_EVALUATE = ALL_LANDMARKS
LANDMARKS_TO_EVALUATE = LANDMARKS_NO_OUTLINE
# LANDMARKS_TO_EVALUATE = range(17, 27)

COARSE_LANDMARKS = range(68)
# COARSE_LANDMARKS = [8, 36, 45, 48, 54]
COARSE_LANDMARKS_TO_ID = {lm: i for i,lm in enumerate(COARSE_LANDMARKS)}

LANDMARK_ID_TO_HEATMAP_ID = {lm: i for i,lm in enumerate(LANDMARKS)}
LANDMARK_HEATMAPS = len(LANDMARKS)

