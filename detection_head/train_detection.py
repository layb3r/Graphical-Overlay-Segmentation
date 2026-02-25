import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for graphical overlay segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m.pt",
        help="Path to YOLO model weights or model name"
    )
    
    # Data configuration
    parser.add_argument(
        "--data",
        type=str,
        default="./data/GOoNS/goons.yaml",
        help="Path to dataset YAML configuration file"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[256, 448],
        help="Image size for training (height width) or single value for square"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto batch)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for training (e.g., '0' or '0,1' or 'cpu')"
    )
    
    parser.add_argument(
        "--rect",
        action="store_true",
        default=True,
        help="Use rectangular training"
    )
    
    parser.add_argument(
        "--no-rect",
        dest="rect",
        action="store_false",
        help="Disable rectangular training"
    )
    
    # Optimization parameters
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor"
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer to use"
    )
    
    # Augmentation parameters
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable data augmentation"
    )
    
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable data augmentation"
    )
    
    # Output configuration
    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing experiment"
    )
    
    # Advanced options
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save training checkpoints"
    )
    
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (disabled if -1)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("YOLO Graphical Overlay Segmentation Training")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Rectangular training: {args.rect}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 70)
    
    # Verify data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {args.data}")
    
    # Initialize model
    print(f"\nInitializing YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Prepare training arguments
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "rect": args.rect,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "optimizer": args.optimizer,
        "augment": args.augment,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "save": args.save,
        "save_period": args.save_period,
        "workers": args.workers,
        "resume": args.resume,
        "pretrained": args.pretrained,
        "verbose": args.verbose,
    }
    
    # Add device if specified
    if args.device:
        train_args["device"] = args.device
    
    # Start training
    print("\nStarting training...\n")
    results = model.train(**train_args)
    
    # Print training summary
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best weights saved to: {model.trainer.best}")
    print(f"Last weights saved to: {model.trainer.last}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()