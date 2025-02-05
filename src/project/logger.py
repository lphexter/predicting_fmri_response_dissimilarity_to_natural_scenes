"""App logger."""

import logging

# Create a named logger
logger = logging.getLogger("predicting_fmri_response_dissimilarity_to_natural_scenes")
logger.setLevel(logging.INFO)  # Set global logging level

# Prevent log duplication from root logger
logger.propagate = False

if not logger.handlers:
    # File Handler - Logs Everything
    file_handler = logging.FileHandler("main.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s")
    )

    # Stream Handler - Logs Only Warnings and Errors
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
