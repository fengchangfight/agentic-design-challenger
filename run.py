import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import uvicorn
from src.config import load_config

if __name__ == "__main__":
    config = load_config()
    server = config["server"]
    uvicorn.run(
        "src.main:app",
        host=server["host"],
        port=server["port"],
        reload=True
    )
