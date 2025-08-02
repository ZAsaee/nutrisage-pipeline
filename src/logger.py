# src/logger.py
import logging

# Configure root logger once for the whole package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# import this named logger elsewhere
logger = logging.getLogger("nutrisage")
