from os import getenv

HOME_DIR = getenv('HOME', "/home/{}".format(getenv('USER', "/home/amirf")))
CAUSALM_DIR = f"{HOME_DIR}/GoogleDrive/AmirNadav/CausaLM"
DATA_DIR = f"{CAUSALM_DIR}/Data"
EXPERIMENTS_DIR = f"{CAUSALM_DIR}/Experiments"
SENTIMENT_DATA_DIR = f"{DATA_DIR}/Sentiment"
SENTIMENT_EXPERIMENTS_DIR = f"{EXPERIMENTS_DIR}/Sentiment"
SENTIMENT_RAW_DATA_DIR = f"{SENTIMENT_DATA_DIR}/Raw"
AMAZON_DATA_DIR = f"{DATA_DIR}/Amazon"
MOVIES_DATA_DIR = f"{SENTIMENT_RAW_DATA_DIR}/movies/"
SENTIMENT_DOMAINS = ("movies", "books", "electronics", "kitchen", "dvd")
SENTIMENT_MODES = ["IMA", "MLM", "OOB"]
DOMAIN = "movies"
MODE = "OOB"
BERT_PRETRAINED_MODEL = 'bert-base-cased'
MAX_SENTIMENT_SEQ_LENGTH = 384
MAX_POMS_SEQ_LENGTH = 12
RANDOM_SEED = 212
NUM_CPU = 4
NUM_GPU = 1
SENTIMENT_IMA_DATA_DIR = f"{DATA_DIR}/Sentiment/IMA"
SENTIMENT_MLM_DATA_DIR = f"{DATA_DIR}/Sentiment/MLM"
IMA_PRETRAINED_MODEL = f"{SENTIMENT_IMA_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
MLM_PRETRAINED_MODEL = f"{SENTIMENT_MLM_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
OOB_PRETRAINED_MODEL = BERT_PRETRAINED_MODEL
if MODE == "OOB":
    FINAL_PRETRAINED_MODEL = OOB_PRETRAINED_MODEL
POMS_DATA_DIR = f"{DATA_DIR}/POMS"
POMS_RAW_DATA_DIR = f"{POMS_DATA_DIR}/Equity-Evaluation-Corpus"
POMS_GENDER_DATA_DIR = f"{POMS_DATA_DIR}/Gender"
POMS_GENDER_DATASETS_DIR = f"{POMS_DATA_DIR}/Gender/Datasets"
POMS_MLM_DATA_DIR = f"{POMS_DATA_DIR}/MLM"
POMS_PRETRAIN_DATA_DIR = f"{POMS_DATA_DIR}/Pretrain"
POMS_EXPERIMENTS_DIR = f"{EXPERIMENTS_DIR}/POMS"
