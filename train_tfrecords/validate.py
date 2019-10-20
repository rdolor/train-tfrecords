import logging
import config as config

INI_SECTIONS     = ["DATA", "MODEL", "TRAINING", "DIRECTORY"]
DATA_FIELDS      = ["train_data_path","start_date","train_period","to_ohe","random_seed","num_cores","train_ratio","valid_ratio","batch_size","prefetch_size"]
#MODEL_FIELDS     = ["DMP_LABEL_COUNT", "CLIENT_LABEL_COUNT"]
#TRAINING_FIELDS  = ["DMP_LABEL_COUNT", "CLIENT_LABEL_COUNT"]
#DIRECTORY_FIELDS = ["DMP_LABEL_COUNT", "CLIENT_LABEL_COUNT"]


["train_data_path","start_date","train_period","to_ohe","random_seed","num_cores","train_ratio","valid_ratio","batch_size","prefetch_size"]

def config_validate():
    valid = []
    
    # Validate sections
    section_diff = list(set(INI_SECTIONS)-set(config.INI.sections()))
    if len(section_diff)>0:
        logging.error("ini file validation 'FAIL', missing sections %s", section_diff)
        valid.append(False)

    # Validate DATA section fields
    agg_dict = dict(config.INI.items("DATA"))
    if len(agg_dict)==0:
        logging.error("ini file validation 'FAIL', missing fields 'AGGREGATION' section.")
        valid.append(False)
    for key in DATA_FIELDS:
        if key not in agg_dict:
            logging.error("ini file validation 'FAIL', missing fields '%s' under 'AGGREGATION' section.", key)
            valid.append(False)
    
    if len(valid) > 0:
        logging.error("ini file validation 'FAIL'.")
        return False

    logging.info("ini file validation 'PASS', will start aggregation process.")
    return True