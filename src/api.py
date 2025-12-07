"""
FastAPI application for video segmentation analysis.
"""
import os
import glob
import yaml
import csv
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys

from src.utils.model_loader import load_models, get_device
from src.utils.video_processing import find_segment_boundaries, analyze_segment_frames
from src.utils.image_processing import (
    find_images,
    load_image,
    preprocess_image_for_segmentation,
    compute_mask,
    resize_mask_to_original,
    save_mask,
    save_mask_overlay,
    get_mask_statistics
)


app = FastAPI(
    title="Video Segmentation API",
    description="API for analyzing video segments using classifier and segmentation models",
    version="1.0.0"
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for models and config
config = None
classifier_model = None
segmentation_model = None
device = None

# Store the 10 most recent measurements
measurements_history: List[Dict[str, Any]] = []
MAX_MEASUREMENTS = 10


def load_config():
    """Load configuration from config.yml"""
    global config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_models():
    """Load the classifier and segmentation models"""
    global classifier_model, segmentation_model, device, config
    
    if config is None:
        load_config()
    
    # Get device and load models
    device = get_device(config)
    classifier_model, segmentation_model = load_models(config, device)


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_config()
    setup_models()




@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Segmentation API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze_video",
            "/measurements",
            "/compute_masks",
            "/health"
        ]
    }


def run_video_analysis_task(
    video_path: str,
    num_segments: int,
    high_conf_threshold: float,
    low_conf_threshold: float,
    num_bottom_rows: int,
    target_size: tuple
):
    """
    Background task to run video analysis.
    This runs asynchronously so it doesn't block the API.
    """
    print(f"\n{'='*80}")
    print(f"[VIDEO ANALYSIS STARTED] Requested segments: {num_segments}")
    print(f"[VIDEO ANALYSIS STARTED] Video: {video_path}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    try:
        # Step 1: Find segment boundaries
        print(f"[PHASE 1/2] Finding segment boundaries...")
        sys.stdout.flush()
        
        segment_boundaries = find_segment_boundaries(
            video_path,
            num_segments,
            high_conf_threshold,
            low_conf_threshold,
            target_size,
            classifier_model,
            device
        )
        
        # Check if we found enough boundaries
        if len(segment_boundaries) < num_segments:
            print(f"[WARNING] Only found {len(segment_boundaries)} boundaries, need {num_segments}")
            sys.stdout.flush()
            return
        
        if len(segment_boundaries) == num_segments:
            print(f"[WARNING] Found segment starts but not end boundary")
            sys.stdout.flush()
            return
        
        # Step 2: Analyze each segment
        actual_segments = len(segment_boundaries) - 1
        print(f"\n[PHASE 2/2] Analyzing {actual_segments} segments...")
        sys.stdout.flush()
        
        results = analyze_segment_frames(
            video_path,
            segment_boundaries,
            target_size,
            num_bottom_rows,
            segmentation_model,
            device
        )
        
        # Create measurement object
        measurement = {
            "id": len(measurements_history) + 1,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "video_path": video_path,
            "num_segments_requested": num_segments,
            "num_segments_found": len(results),
            "configuration": {
                "intersection_confidence_threshold": high_conf_threshold,
                "intersection_no_confidence_threshold": low_conf_threshold,
                "segmentation_rows": num_bottom_rows,
                "target_size": list(target_size)
            },
            "segments": results
        }
        
        # Save results to CSV file
        csv_dir = os.path.join("output", "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_filename = f"{video_name}_segments_{timestamp_str}.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['segment_id', 'x_min', 'x_max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for segment in results:
                writer.writerow({
                    'segment_id': segment['segment'] - 1,
                    'x_min': segment['min_x'] if segment['min_x'] is not None else '',
                    'x_max': segment['max_x'] if segment['max_x'] is not None else ''
                })
        
        print(f"\n[CSV SAVED] {csv_path}")
        sys.stdout.flush()
        
        measurement['csv_path'] = csv_path
        
        # Store measurement
        measurements_history.append(measurement)
        if len(measurements_history) > MAX_MEASUREMENTS:
            measurements_history.pop(0)
        
        print(f"\n{'='*80}")
        print(f"[VIDEO ANALYSIS COMPLETE] Measurement #{measurement['id']} stored")
        print(f"[VIDEO ANALYSIS COMPLETE] Found {len(results)} segments")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"\n[ERROR] Video analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


@app.get("/analyze_video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    num_segments: int = Query(..., description="Number of segments to find in the video", gt=0)
):
    """
    Analyze video segments and return min/max x coordinates for each segment.
    
    Args:
        num_segments: Number of segments to find
        
    Returns:
        JSON with segment analysis results
    """
    global config
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    # Find video file
    videos_dir = config["api"]["videos_dir"]
    
    if not os.path.exists(videos_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Videos directory not found: {videos_dir}"
        )
    
    # Look for video.mp4 or other video formats
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
    
    # Prioritize video.mp4 if it exists
    video_path = None
    for vf in video_files:
        if os.path.basename(vf).startswith("video."):
            video_path = vf
            break
    
    if video_path is None and video_files:
        video_path = video_files[0]  # Use first available video
    
    if video_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No video file found in {videos_dir}"
        )
    
    print(f"[REQUEST] Video analysis requested for {num_segments} segments")
    print(f"[REQUEST] Video: {video_path}")
    sys.stdout.flush()
    
    # Get configuration parameters
    high_conf_threshold = config["api"]["intersection_confidence_threshold"]
    low_conf_threshold = config["api"]["intersection_no_confidence_threshold"]
    num_bottom_rows = config["api"]["segmentation_rows"]
    target_size = tuple(config["data"]["target_size"])
    
    # Schedule the analysis to run in the background
    background_tasks.add_task(
        run_video_analysis_task,
        video_path,
        num_segments,
        high_conf_threshold,
        low_conf_threshold,
        num_bottom_rows,
        target_size
    )
    
    # Return immediately with accepted status
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": f"Video analysis started for {num_segments} segments",
            "video_path": video_path,
            "note": "Processing in background. Check /measurements endpoint for results when complete."
        }
    )


@app.get("/measurements")
async def get_measurements():
    """
    Get the 10 most recent measurements.
    
    Returns:
        JSON array of recent measurements (most recent first)
    """
    # Return in reverse order (most recent first)
    return JSONResponse(
        status_code=200,
        content={
            "count": len(measurements_history),
            "measurements": list(reversed(measurements_history))
        }
    )


@app.get("/compute_masks")
async def compute_masks(
    save_overlays: bool = Query(True, description="Whether to save overlay images"),
    return_statistics: bool = Query(True, description="Whether to return mask statistics")
):
    """
    Compute segmentation masks for all images in the images directory.
    
    This endpoint:
    1. Finds all images in the configured images directory
    2. Processes each image with the segmentation model
    3. Saves binary masks to the output directory
    4. Optionally saves overlay images
    5. Returns statistics about each mask
    
    Args:
        save_overlays: Whether to save images with mask overlays
        return_statistics: Whether to compute and return statistics
        
    Returns:
        JSON with processing results for each image
    """
    global config, segmentation_model, device
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    if segmentation_model is None:
        raise HTTPException(status_code=500, detail="Segmentation model not loaded")
    
    # Get directories from config
    images_dir = config["api"]["images_dir"]
    masks_output_dir = config["api"]["masks_output_dir"]
    target_size = tuple(config["data"]["target_size"])
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Images directory not found: {images_dir}"
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(masks_output_dir, exist_ok=True)
    if save_overlays:
        overlays_dir = os.path.join(masks_output_dir, "overlays")
        os.makedirs(overlays_dir, exist_ok=True)
    
    # Find all images
    image_files = find_images(images_dir)
    
    if not image_files:
        raise HTTPException(
            status_code=404,
            detail=f"No image files found in {images_dir}"
        )
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Output directory: {masks_output_dir}")
    
    results = []
    
    try:
        for idx, image_path in enumerate(image_files):
            image_name = os.path.basename(image_path)
            print(f"Processing [{idx+1}/{len(image_files)}]: {image_name}")
            
            try:
                # Load image
                image = load_image(image_path)
                
                # Preprocess for segmentation
                image_tensor, original_size = preprocess_image_for_segmentation(
                    image, target_size
                )
                
                # Compute mask at model resolution
                mask = compute_mask(image_tensor, segmentation_model, device)
                
                # Resize mask back to original size
                mask_original = resize_mask_to_original(mask, original_size)
                
                # Generate output filenames
                name_without_ext = os.path.splitext(image_name)[0]
                mask_path = os.path.join(masks_output_dir, f"{name_without_ext}_mask.png")
                
                # Save binary mask
                save_mask(mask_original, mask_path)
                
                # Optionally save overlay
                overlay_path = None
                if save_overlays:
                    overlay_path = os.path.join(overlays_dir, f"{name_without_ext}_overlay.png")
                    save_mask_overlay(image, mask_original, overlay_path)
                
                # Compute statistics
                stats = None
                if return_statistics:
                    stats = get_mask_statistics(mask_original)
                
                result = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "original_size": {"height": original_size[0], "width": original_size[1]},
                    "status": "success"
                }
                
                if stats:
                    result["statistics"] = stats
                
                results.append(result)
                print(f"  ✓ Saved mask to: {mask_path}")
                
            except Exception as e:
                error_result = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "status": "error",
                    "error": str(e)
                }
                results.append(error_result)
                print(f"  ✗ Error processing {image_name}: {e}")
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
        
        return JSONResponse(
            status_code=200,
            content={
                "images_directory": images_dir,
                "output_directory": masks_output_dir,
                "total_images": len(results),
                "successful": successful,
                "failed": failed,
                "results": results
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during mask computation: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": classifier_model is not None and segmentation_model is not None,
        "device": str(device) if device else None
    }


if __name__ == "__main__":
    import uvicorn
    
    load_config()
    
    host = config["api"]["host"]
    port = config["api"]["port"]
    
    print(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

