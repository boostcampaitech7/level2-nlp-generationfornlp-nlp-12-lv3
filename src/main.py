import argparse
import json

def get_model_name(config):
    with open(config + ".json", "r") as f:
        config = json.load(f)
    return config["model"]["name"]

def main(Pipeline, Manager):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/baseline", help="path where config json is store")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--inference", action="store_true", help="inference the model")
    parser.add_argument("--do_both", action="store_true", help="train and inference the model")
    args = parser.parse_args()

    if not args.train and not args.inference and not args.do_both:
        print("No action specified. Do train.")
        args.train = True 
    
    if args.do_both:
        args.train = True
        args.inference = True
    
    pipeline = Pipeline(args.config, Manager)
    
    if args.train:
        pipeline.train()
    if args.inference:
        if not args.train and not "checkpoint" in get_model_name(args.config):
            print("! ! ! No checkpoint found. Vanilla model will be used for inference. ! ! !")
        pipeline.inference()
        
if __name__ == "__main__":
    from pipeline import BasePipeline
    from manager import BaselineManager
    main(
        BasePipeline,
        BaselineManager
    )