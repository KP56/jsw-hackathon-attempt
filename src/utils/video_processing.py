"""
Video processing utilities for segment detection and analysis.
"""
import cv2
import numpy as np
import torch
import sys
from typing import List, Dict, Any, Tuple


def preprocess_frame(frame: np.ndarray, target_size: tuple) -> torch.Tensor:
    """
    Preprocess a video frame for model input.
    
    Args:
        frame: BGR frame from OpenCV
        target_size: (height, width) tuple
        
    Returns:
        Preprocessed tensor
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame_resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]))
    
    # Convert to tensor and normalize to [0, 1]
    frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
    
    # Change from HWC to CHW format
    frame_tensor = frame_tensor.permute(2, 0, 1)
    
    return frame_tensor


def get_classifier_confidence(
    frame_tensor: torch.Tensor,
    classifier_model: torch.nn.Module,
    device: torch.device
) -> float:
    """
    Get classifier confidence for a frame.
    
    Args:
        frame_tensor: Preprocessed frame tensor
        classifier_model: Trained classifier model
        device: Device to run inference on
        
    Returns:
        Confidence score (sigmoid of logit)
    """
    with torch.no_grad():
        frame_batch = frame_tensor.unsqueeze(0).to(device)
        logit = classifier_model(frame_batch)
        confidence = torch.sigmoid(logit).item()
    return confidence


def get_segmentation_mask(
    frame_tensor: torch.Tensor,
    segmentation_model: torch.nn.Module,
    device: torch.device
) -> np.ndarray:
    """
    Get segmentation mask for a frame.
    
    Args:
        frame_tensor: Preprocessed frame tensor
        segmentation_model: Trained segmentation model
        device: Device to run inference on
        
    Returns:
        Binary segmentation mask
    """
    with torch.no_grad():
        frame_batch = frame_tensor.unsqueeze(0).to(device)
        logits = segmentation_model(frame_batch)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def find_segment_boundaries(
    video_path: str,
    num_segments: int,
    high_conf_threshold: float,
    low_conf_threshold: float,
    target_size: tuple,
    classifier_model: torch.nn.Module,
    device: torch.device
) -> List[int]:
    """
    Find frame indices where new segments start.
    
    We find num_segments + 1 boundaries to properly determine where the last
    requested segment ends (trim to required length).
    
    Args:
        video_path: Path to video file
        num_segments: Number of segments to find
        high_conf_threshold: Threshold for high confidence (intersection)
        low_conf_threshold: Threshold for low confidence (no intersection)
        target_size: Target size for frame preprocessing
        classifier_model: Trained classifier model
        device: Device to run inference on
        
    Returns:
        List of frame indices where segments start (length: num_segments + 1, or less if video ends)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Boundary Detection] Video has {total_frames} total frames")
    print(f"[Boundary Detection] Looking for {num_segments + 1} boundaries...")
    sys.stdout.flush()
    
    segment_start_frames = []
    frame_idx = 0
    waiting_for_low_conf = False
    log_interval = max(1, total_frames // 20)  # Log every 5% of video
    
    # We need num_segments + 1 boundaries to know where the last segment ends
    required_boundaries = num_segments + 1
    
    try:
        while len(segment_start_frames) < required_boundaries:
            ret, frame = cap.read()
            
            if not ret:
                print(f"[Boundary Detection] Reached end of video at frame {frame_idx}")
                break  # End of video
            
            # Preprocess frame
            frame_tensor = preprocess_frame(frame, target_size)
            
            # Get classifier confidence
            confidence = get_classifier_confidence(frame_tensor, classifier_model, device)
            
            if not waiting_for_low_conf:
                # Looking for high confidence (intersection)
                if confidence >= high_conf_threshold:
                    waiting_for_low_conf = True
                    print(f"[Boundary Detection] Frame {frame_idx}: High confidence detected ({confidence:.3f})")
                    sys.stdout.flush()
            else:
                # Waiting for low confidence (segment start)
                if confidence <= low_conf_threshold:
                    segment_start_frames.append(frame_idx)
                    print(f"[Boundary Detection] Frame {frame_idx}: Segment boundary #{len(segment_start_frames)} found! ({confidence:.3f})")
                    sys.stdout.flush()
                    waiting_for_low_conf = False
            
            # Log progress periodically
            if frame_idx % log_interval == 0 and frame_idx > 0:
                progress = (frame_idx / total_frames) * 100
                print(f"[Boundary Detection] Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%) - Found {len(segment_start_frames)} boundaries")
                sys.stdout.flush()
            
            frame_idx += 1
    
    finally:
        cap.release()
        print(f"[Boundary Detection] Complete! Found {len(segment_start_frames)} boundaries")
        sys.stdout.flush()
    
    return segment_start_frames


def analyze_segment_frames(
    video_path: str,
    segment_boundaries: List[int],
    target_size: tuple,
    num_bottom_rows: int,
    segmentation_model: torch.nn.Module,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Analyze frames in each segment and compute min/max x coordinates.
    
    Expects segment_boundaries to have length num_segments + 1, where the last
    boundary marks the end of the final segment (video is trimmed to this length).
    
    Args:
        video_path: Path to video file
        segment_boundaries: List of frame indices where segments start (length: num_segments + 1)
        target_size: Target size for frame preprocessing
        num_bottom_rows: Number of bottom rows to analyze
        segmentation_model: Trained segmentation model
        device: Device to run inference on
        
    Returns:
        List of segment analysis results (length: num_segments)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # We process num_segments (not including the last boundary which is just the endpoint)
    num_segments = len(segment_boundaries) - 1
    print(f"[Segment Analysis] Starting analysis of {num_segments} segments")
    sys.stdout.flush()
    
    results = []
    
    try:
        for seg_idx in range(num_segments):
            start_frame = segment_boundaries[seg_idx]
            end_frame = segment_boundaries[seg_idx + 1]
            num_frames_in_segment = end_frame - start_frame
            
            print(f"[Segment Analysis] Segment {seg_idx + 1}/{num_segments}: Frames {start_frame}-{end_frame - 1} ({num_frames_in_segment} frames)")
            sys.stdout.flush()
            
            segment_min_x = None
            segment_max_x = None
            
            # Process frames in this segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_processed = 0
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                
                if not ret:
                    print(f"[Segment Analysis] Warning: Could not read frame {frame_idx}")
                    break
                
                # Preprocess and get segmentation mask
                frame_tensor = preprocess_frame(frame, target_size)
                mask = get_segmentation_mask(frame_tensor, segmentation_model, device)
                
                # Get bottom rows
                bottom_rows = mask[-num_bottom_rows:, :]
                
                # Find x coordinates of segmented pixels in bottom rows
                segmented_pixels = np.where(bottom_rows > 0)
                
                if len(segmented_pixels[1]) > 0:  # If there are segmented pixels
                    frame_min_x = np.min(segmented_pixels[1])
                    frame_max_x = np.max(segmented_pixels[1])
                    
                    # Update segment min/max
                    if segment_min_x is None or frame_min_x < segment_min_x:
                        segment_min_x = frame_min_x
                    if segment_max_x is None or frame_max_x > segment_max_x:
                        segment_max_x = frame_max_x
                
                frames_processed += 1
                
                # Log progress every 10 frames or for small segments, every frame
                if num_frames_in_segment <= 10 or frames_processed % 10 == 0 or frames_processed == num_frames_in_segment:
                    frames_left = num_frames_in_segment - frames_processed
                    progress = (frames_processed / num_frames_in_segment) * 100
                    print(f"[Segment Analysis]   Segment {seg_idx + 1}: Frame {frame_idx} processed ({frames_processed}/{num_frames_in_segment}, {progress:.1f}%) - {frames_left} frames left")
                    sys.stdout.flush()
            
            result = {
                "segment": seg_idx + 1,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame - 1),
                "min_x": int(segment_min_x) if segment_min_x is not None else None,
                "max_x": int(segment_max_x) if segment_max_x is not None else None
            }
            results.append(result)
            print(f"[Segment Analysis] Segment {seg_idx + 1} complete: min_x={segment_min_x}, max_x={segment_max_x}")
            sys.stdout.flush()
    
    finally:
        cap.release()
        print(f"[Segment Analysis] All {num_segments} segments analyzed successfully!")
        sys.stdout.flush()
    
    return results

