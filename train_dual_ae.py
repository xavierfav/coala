import json
import argparse

from dual_ae_trainer import DualAETrainer


def main(config_file):
    params = json.load(open(config_file, 'rb'))
    print("Training Dual AutoEncoder with params:")
    print(json.dumps(params, separators=("\n", ": "), indent=4))
    trainer = DualAETrainer(params)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dual Auto Encoder')
    parser.add_argument('config_file', type=str,
                        help='configuration file for the training')         
    args = parser.parse_args()

    main(args.config_file)
