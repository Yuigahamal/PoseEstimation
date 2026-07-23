import os

from mmengine.config import Config
from mmengine.runner import Runner
from VitPose.configs import ParamConfig
from VitPose.models import VIT,KeyPointLoss



def main():
    # ParamConfig.set_env()
    cfg = Config.fromfile(ParamConfig.CONFIG_DIR/'config.py')
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()