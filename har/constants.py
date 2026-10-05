from __future__ import annotations

NUM_CLASSES = 40
NUM_TEST_CLIPS = 405
NUM_SKELETON_JOINTS = 17
NUM_IMU_SENSORS = 5
IMU_FEATS_PER_SENSOR = 9

MASK_MODALITIES = ("depth", "ir", "thermal", "imu", "skeleton", "radar")

NUM_FRAMES = 8
IMAGE_SIZE = 64
IMU_LENGTH = 128
SKELETON_LENGTH = 32
RADAR_LENGTH = 32

TEST_ID_FMT = "SM_test_{idx:04d}"
TEST_PATH_PREFIX = "small_model_track_test"

SUBMISSION_COLUMNS = ("path", "prediction")

CLASS_NAMES = [
    "0_Wash_face",
    "1_Brush_teeth",
    "2_Comb_hair",
    "3_Take_off_clothes",
    "4_Wipe_hands",
    "5_Put_on_clothes",
    "6_Drink_water",
    "7_Eat_food",
    "8_Take_and_use_tableware",
    "9_Pour_drinks",
    "10_Stir_drinks",
    "11_Peel_fruits",
    "12_Sweep_the_floor",
    "13_Mop_the_floor",
    "14_Wipe_bowls",
    "15_Wipe_windows_and_tables",
    "16_Fold_clothes",
    "17_Tap_the_keyboard",
    "18_Write",
    "19_Make_a_phone_call",
    "20_Check_the_time",
    "21_Read_documents",
    "22_Turn_pages",
    "23_Listen_to_music_with_headphones",
    "24_Use_a_mobile_phone",
    "25_Watch_TV",
    "26_Play_games",
    "27_Take_a_selfie",
    "28_Jog_in_place",
    "29_Do_squats",
    "30_Do_jumping_jacks",
    "31_Do_stretching_exercises",
    "32_Stand_up",
    "33_Lie_down",
    "34_Sit_down",
    "35_Do_lunges",
    "36_Walk",
    "37_Take_medicine",
    "38_Massage_oneself",
    "39_Take_body_temperature",
]

MODALITY_ALIASES = {
    "depth": ("Depth_Color", "Depth", "depth_color", "depth", "DepthColor"),
    "ir": ("IR", "ir", "Infrared", "infrared"),
    "thermal": ("Thermal", "thermal", "THERMAL"),
    "imu": ("IMU", "imu", "Imu"),
    "radar": ("Radar", "radar", "mmWave", "mmwave", "MMWave"),
    "skeleton": ("Skeleton", "skeleton", "Pose", "pose", "Keypoints"),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_EXTS = {".csv", ".txt", ".tsv"}
ARRAY_EXTS = {".npy", ".npz", ".pkl", ".pickle", ".json"}

TRAIN_USERS = tuple(list(range(1, 10)) + list(range(16, 25)))
TEST_USERS = (10, 11, 25, 26)
