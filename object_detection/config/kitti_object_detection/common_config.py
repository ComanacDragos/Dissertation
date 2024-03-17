from pathlib import Path
from backend.utils import set_seed
import shutil


class CommonConfig:
    EXPERIMENT = Path("outputs/kitti")

    @staticmethod
    def init_experiment():
        set_seed(0)
        shutil.copytree('config', CommonConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
