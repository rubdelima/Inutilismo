import yaml
import multiprocessing as mp
import torch

with open('config.yaml', encoding='utf-8') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

if torch.cuda.is_available():
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

NUM_WORKERS = min(GLOBAL_CONFIG.get("memory_manager", {}).get("min_num_workers", 4), mp.cpu_count() - 1)
