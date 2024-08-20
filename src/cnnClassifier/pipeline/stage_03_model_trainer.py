from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import TrainningConfig
from cnnClassifier.components.model_training import Trainning
from cnnClassifier import logger


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Trainning(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ == "__main__":
    try:
        logger.info(f"*****************")
        logger.info(f">> Stage {STAGE_NAME} started <<--<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<<<")
    
    except Exception as e:
        logger.exception(e)
        raise e

