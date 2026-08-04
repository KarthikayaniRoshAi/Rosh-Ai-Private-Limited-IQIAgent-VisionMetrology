from dotenv import load_dotenv

load_dotenv()

from common.config_loader import ConfigLoader
from common.logger import logger
from common.exceptions import InvalidDomainException

def print_banner(config, domain, mode):
    logger.info("")
    logger.info("============================================================")
    logger.info("                  IQI™ Framework")
    logger.info("       Intelligent Quality Inspection Platform")
    logger.info("------------------------------------------------------------")
    logger.info(f" Application        : {config['application']['name']}")
    logger.info(f" Version            : {config['application']['version']}")
    logger.info(f" Engine             : {domain.replace('_', ' ').title()}")
    logger.info(f" Mode               : {mode.title()}")
    logger.info(" Configuration      : configs/main_config.yaml")
    logger.info(" Status             : Initialized")
    logger.info("============================================================")
    logger.info("")

def main():
    config = ConfigLoader("configs/main_config.yaml").get()
    domain = config["engine"]["domain"]
    mode = config["engine"]["mode"]

    print_banner(config, domain, mode)

    logger.info("Loading Framework...")

    if domain == "visual_inspection":
        from visual_inspection import engine

        logger.info("✓ Visual Inspection Engine Loaded")

        if mode.lower() == "train":
            logger.info("Starting Training...")
            engine.train()
        elif mode.lower() == "test":
            logger.info("Starting Testing...")
            engine.test()
        else:
            raise InvalidDomainException(
                f"Unsupported visual inspection mode: {mode}"
            )

    elif domain == "visual_metrology":
        from visual_metrology import engine

        logger.info("✓ Visual Metrology Engine Loaded")

        vm_config = ConfigLoader(
            "visual_metrology/configs/config.yaml"
        ).get()

        engine.run(vm_config)

    else:
        raise InvalidDomainException(
            f"Unsupported inspection domain: {domain}"
        )

if __name__ == "__main__":
    main()











# from dotenv import load_dotenv

# load_dotenv()

# from common.config_loader import ConfigLoader
# from common.logger import logger
# from common.exceptions import InvalidDomainException
# from visual_inspection import engine

# def print_banner(config, domain, mode):
#     logger.info("")
#     logger.info("============================================================")
#     logger.info("                 IQI™ Framework")
#     logger.info("      Intelligent Quality Inspection Platform")
#     logger.info("------------------------------------------------------------")
#     logger.info(f" Application        : {config['application']['name']}")
#     logger.info(f" Version            : {config['application']['version']}")
#     logger.info(f" Engine             : {domain.replace('_', ' ').title()}")
#     logger.info(f" Mode               : {mode.title()}")
#     logger.info(" Configuration      : configs/main_config.yaml")
#     logger.info(" Status             : Initialized")
#     logger.info("============================================================")
#     logger.info("")

# def main():

#     config = ConfigLoader("configs/main_config.yaml").get()
#     domain = config["engine"]["domain"]
#     mode = config["engine"]["mode"]

#     print_banner(config, domain, mode)

#     logger.info("Loading Framework...")

#     if domain == "visual_inspection":

#         from visual_inspection.engine import train, test

#         logger.info("✓ Visual Inspection Engine Loaded")

#         if mode.lower() == "train":

#             logger.info("Starting Training...")
#             engine.train()

#         elif mode.lower() == "test":

#             logger.info("Starting Testing...")
#             engine.test()

#         else:

#             raise InvalidDomainException(
#                 f"Unsupported visual inspection mode: {mode}"
#             )

#     elif domain == "visual_metrology":

#         from visual_metrology import engine

#         logger.info("✓ Visual Metrology Engine Loaded")

#         vm_config = ConfigLoader(
#             "visual_metrology/configs/config.yaml"
#         ).get()

#         engine.run(vm_config)

#     else:

#         raise InvalidDomainException(
#             f"Unsupported inspection domain: {domain}"
#         )


# if __name__ == "__main__":
#     main()