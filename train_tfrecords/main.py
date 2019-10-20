import config
import validate
import utils

log = utils.create_logger()

if validate.config_validate():
    log.info(config.INI['TRAINING']['new_run'])
    log.info('OK')