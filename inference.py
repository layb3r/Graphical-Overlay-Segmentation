import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run combined YOLO + FastSAM inference for graphical overlay segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="./weights/yolo_best.pt",
        help="Path to trained YOLO detection model weights (.pt file)"
    )
    
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default="./weights/FastSAM-x.pt",
        help="Path to FastSAM model weights (.pt file)"
    )
    
    # Input configuration
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (image file, video file, directory, or camera index)"
    )
    
    # Inference parameters
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (FastSAM typically uses 1024)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO detections"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="IoU threshold for NMS in FastSAM"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for inference (e.g., '0' or 'cpu')"
    )
    
    parser.add_argument(
        "--retina",
        action="store_true",
        help="Use retina masks for higher quality FastSAM output"
    )
    
    # Visualization options
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save inference results"
    )
    
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Don't save inference results"
    )
    
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save individual mask files"
    )
    
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save overlay visualization"
    )
    
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width (pixels)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparency for mask overlay (0.0-1.0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--project",
        type=str,
        default="runs/combined_inference",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing experiment"
    )
    
    # Advanced options
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum number of detections per image"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Show timing information for each stage"
    )
    
    return parser.parse_args()


def get_boxes_from_yolo(yolo_model, image, conf_threshold, device):
    """
    Run YOLO detection to get bounding boxes.
    
    Args:
        yolo_model: YOLO model instance
        image: Input image (numpy array)
        conf_threshold: Confidence threshold
        device: Device to run on
        
    Returns:
        boxes: Detected bounding boxes in xyxy format
        confidences: Confidence scores
        class_ids: Class IDs
    """
    # Run YOLO detection
    results = yolo_model.predict(
        image,
        conf=conf_threshold,
        device=device,
        verbose=False
    )
    
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None, None
    
    # Extract boxes, confidences, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    return boxes, confidences, class_ids


def run_fastsam_with_boxes(fastsam_model, image, boxes, iou_threshold, device, retina=False):
    """
    Run FastSAM with box prompts from YOLO detections.
    
    Args:
        fastsam_model: FastSAM model instance
        image: Input image (numpy array)
        boxes: Bounding boxes from YOLO in xyxy format
        iou_threshold: IoU threshold for FastSAM
        device: Device to run on
        retina: Use retina masks
        
    Returns:
        masks: Segmentation masks
        fastsam_results: Full FastSAM results
    """
    # Run FastSAM
    fastsam_results = fastsam_model(
        image,
        device=device,
        retina_masks=retina,
        iou=iou_threshold,
        conf=0.4,
        verbose=False
    )
    
    if len(fastsam_results) == 0:
        return None, None
    
    # Create prompt processor
    prompt_process = FastSAMPrompt(image, fastsam_results, device=device)
    
    # Use box prompts
    # Convert boxes to list format expected by FastSAM
    box_prompts = boxes.tolist() if len(boxes) > 0 else None
    
    if box_prompts is None:
        return None, None
    
    # Get masks using box prompts
    masks = prompt_process.box_prompt(bboxes=box_prompts)
    
    return masks, fastsam_results


def visualize_results(image, boxes, masks, confidences, class_ids, class_names, 
                     alpha=0.4, line_width=2):
    """
    Visualize combined YOLO + FastSAM results.
    
    Args:
        image: Original image
        boxes: YOLO bounding boxes
        masks: FastSAM segmentation masks
        confidences: Detection confidences
        class_ids: Class IDs
        class_names: Class names dictionary
        alpha: Mask transparency
        line_width: Box line width
        
    Returns:
        vis_image: Visualization image
    """
    vis_image = image.copy()
    
    # Generate colors for each detection
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
    
    # Apply masks
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            if i >= len(colors):
                break
                
            color = colors[i].tolist()
            
            # Convert mask to binary
            if isinstance(mask, np.ndarray):
                binary_mask = mask.astype(bool)
            else:
                binary_mask = mask.cpu().numpy().astype(bool)
            
            # Apply colored mask
            for c in range(3):
                vis_image[:, :, c] = np.where(
                    binary_mask,
                    vis_image[:, :, c] * (1 - alpha) + alpha * color[c],
                    vis_image[:, :, c]
                )
    
    # Draw bounding boxes and labels
    if boxes is not None and len(boxes) > 0:
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i].tolist()
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_width)
            
            # Draw label
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            label = f"{class_name} {conf:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return vis_image


def process_image(image_path, yolo_model, fastsam_model, args, save_dir=None):
    """
    Process a single image with YOLO + FastSAM.
    
    Args:
        image_path: Path to image file
        yolo_model: YOLO model
        fastsam_model: FastSAM model
        args: CLI arguments
        save_dir: Directory to save results
        
    Returns:
        vis_image: Visualization image
        stats: Processing statistics
    """
    stats = {
        'yolo_time': 0,
        'fastsam_time': 0,
        'total_time': 0,
        'num_detections': 0
    }
    
    start_total = time.time()
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None, stats
    
    original_image = image.copy()
    
    # Step 1: YOLO Detection
    if args.benchmark:
        start = time.time()
    
    boxes, confidences, class_ids = get_boxes_from_yolo(
        yolo_model, image, args.conf, args.device
    )
    
    if args.benchmark:
        stats['yolo_time'] = time.time() - start
    
    if boxes is None or len(boxes) == 0:
        print(f"No detections found in {image_path.name}")
        stats['total_time'] = time.time() - start_total
        return original_image, stats
    
    stats['num_detections'] = len(boxes)
    
    # Step 2: FastSAM with box prompts
    if args.benchmark:
        start = time.time()
    
    masks, fastsam_results = run_fastsam_with_boxes(
        fastsam_model, image, boxes, args.iou, args.device, args.retina
    )
    
    if args.benchmark:
        stats['fastsam_time'] = time.time() - start
    
    # Step 3: Visualize results
    class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
    
    vis_image = visualize_results(
        original_image, boxes, masks, confidences, class_ids,
        class_names, args.alpha, args.line_width
    )
    
    stats['total_time'] = time.time() - start_total
    
    # Save results
    if save_dir and args.save:
        # Save visualization
        if args.save_overlay:
            save_path = save_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(save_path), vis_image)
        
        # Save individual masks
        if args.save_masks and masks is not None:
            masks_dir = save_dir / "masks" / image_path.stem
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            for i, mask in enumerate(masks):
                if isinstance(mask, np.ndarray):
                    mask_img = (mask * 255).astype(np.uint8)
                else:
                    mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                
                mask_path = masks_dir / f"mask_{i}.png"
                cv2.imwrite(str(mask_path), mask_img)
    
    return vis_image, stats


def process_video(video_path, yolo_model, fastsam_model, args, save_dir=None):
    """
    Process a video file with YOLO + FastSAM.
    
    Args:
        video_path: Path to video file
        yolo_model: YOLO model
        fastsam_model: FastSAM model
        args: CLI arguments
        save_dir: Directory to save results
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Setup video writer
    writer = None
    if save_dir and args.save:
        output_path = save_dir / f"{video_path.stem}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            boxes, confidences, class_ids = get_boxes_from_yolo(
                yolo_model, frame, args.conf, args.device
            )
            
            if boxes is not None and len(boxes) > 0:
                total_detections += len(boxes)
                
                # Run FastSAM
                masks, _ = run_fastsam_with_boxes(
                    fastsam_model, frame, boxes, args.iou, args.device, args.retina
                )
                
                # Visualize
                class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
                vis_frame = visualize_results(
                    frame, boxes, masks, confidences, class_ids,
                    class_names, args.alpha, args.line_width
                )
            else:
                vis_frame = frame
            
            # Save frame
            if writer:
                writer.write(vis_frame)
            
            # Show frame
            if args.show:
                cv2.imshow('YOLO + FastSAM Inference', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print(f"Video processing complete: {frame_count} frames, {total_detections} total detections")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("YOLO + FastSAM Combined Inference")
    print("=" * 70)
    print(f"YOLO Model: {args.yolo_model}")
    print(f"FastSAM Model: {args.fastsam_model}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Mask alpha: {args.alpha}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 70)
    
    # Verify model files exist
    yolo_path = Path(args.yolo_model)
    fastsam_path = Path(args.fastsam_model)
    
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model file not found: {args.yolo_model}")
    if not fastsam_path.exists():
        raise FileNotFoundError(f"FastSAM model file not found: {args.fastsam_model}")
    
    # Load models
    print(f"\nLoading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    print("YOLO model loaded!")
    
    print(f"Loading FastSAM model: {args.fastsam_model}")
    fastsam_model = FastSAM(args.fastsam_model)
    print("FastSAM model loaded!")
    
    # Setup save directory
    save_dir = None
    if args.save:
        save_dir = Path(args.project) / args.name
        
        # Handle existing directory
        if save_dir.exists() and not args.exist_ok:
            # Find next available number
            counter = 1
            while (Path(args.project) / f"{args.name}{counter}").exists():
                counter += 1
            save_dir = Path(args.project) / f"{args.name}{counter}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
    
    # Check source type
    source_path = Path(args.source)
    
    # Check if source is camera index
    if args.source.isdigit():
        print("\nCamera input not yet implemented for combined inference")
        return
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    
    # Process based on source type
    print("\nStarting inference...\n")
    
    if source_path.is_file():
        # Check if video or image
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        if source_path.suffix.lower() in video_extensions:
            # Process video
            process_video(source_path, yolo_model, fastsam_model, args, save_dir)
        elif source_path.suffix.lower() in image_extensions:
            # Process single image
            vis_image, stats = process_image(
                source_path, yolo_model, fastsam_model, args, save_dir
            )
            
            if vis_image is not None:
                print(f"\nResults for {source_path.name}:")
                print(f"  Detections: {stats['num_detections']}")
                if args.benchmark:
                    print(f"  YOLO time: {stats['yolo_time']:.3f}s")
                    print(f"  FastSAM time: {stats['fastsam_time']:.3f}s")
                    print(f"  Total time: {stats['total_time']:.3f}s")
                
                if args.show:
                    cv2.imshow('Result', vis_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"Unsupported file format: {source_path.suffix}")
            return
    
    elif source_path.is_dir():
        # Process directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No valid images found in directory!")
            return
        
        print(f"Found {len(image_files)} images")
        
        all_stats = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")
            
            vis_image, stats = process_image(
                img_path, yolo_model, fastsam_model, args, save_dir
            )
            
            if vis_image is not None:
                all_stats.append(stats)
                print(f"  Detections: {stats['num_detections']}")
                if args.benchmark:
                    print(f"  Time: {stats['total_time']:.3f}s")
        
        # Print summary
        if all_stats:
            print("\n" + "=" * 70)
            print("Inference Summary")
            print("=" * 70)
            print(f"Total images processed: {len(all_stats)}")
            print(f"Total detections: {sum(s['num_detections'] for s in all_stats)}")
            if args.benchmark:
                avg_yolo = np.mean([s['yolo_time'] for s in all_stats])
                avg_fastsam = np.mean([s['fastsam_time'] for s in all_stats])
                avg_total = np.mean([s['total_time'] for s in all_stats])
                print(f"Average YOLO time: {avg_yolo:.3f}s")
                print(f"Average FastSAM time: {avg_fastsam:.3f}s")
                print(f"Average total time: {avg_total:.3f}s")
            print("=" * 70)
    
    if save_dir:
        print(f"\nResults saved to: {save_dir}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
