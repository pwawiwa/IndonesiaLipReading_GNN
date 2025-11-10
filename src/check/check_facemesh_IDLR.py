import os
from moviepy import VideoFileClip
from datetime import timedelta
from collections import defaultdict

# --- Configuration ---
ROOT_DIR = "data/IDLR"
OUTPUT_LOG_FILE = "src/reports/IDLR_video_log.txt"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)

def get_video_stats(video_path):
    """Extracts duration, resolution, and file size from a video file."""
    try:
        with VideoFileClip(video_path) as clip:
            duration_s = clip.duration
            resolution = f"{clip.w}x{clip.h}"
            fps = clip.fps
        
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        return {
            "Path": video_path,
            "Duration (s)": duration_s,
            "Resolution": resolution,
            "FPS": fps,
            "Size (MB)": file_size_mb
        }
    except Exception as e:
        return {
            "Path": video_path,
            "Error": f"Could not process video: {e}"
        }

def log_summary_to_file(stats_list, output_file):
    """Calculates and writes summary statistics to the specified file."""
    
    total_videos = len(stats_list)
    successful_videos = [s for s in stats_list if 'Error' not in s]
    num_successful = len(successful_videos)
    num_errors = total_videos - num_successful

    total_duration_s = sum(s['Duration (s)'] for s in successful_videos)
    total_size_mb = sum(s['Size (MB)'] for s in successful_videos)
    
    # Calculate distributions
    resolution_counts = defaultdict(int)
    for s in successful_videos:
        resolution_counts[s['Resolution']] += 1
        
    # Format total duration
    total_duration_formatted = str(timedelta(seconds=round(total_duration_s)))
    
    # Write the summary log
    with open(output_file, 'w') as f:
        f.write("--- 🎬 Dataset Video Summary --- \n\n")
        f.write(f"**Total Files Scanned:** {total_videos}\n")
        f.write(f"**Successfully Processed Videos:** {num_successful}\n")
        f.write(f"**Processing Errors:** {num_errors}\n")
        f.write("-" * 35 + "\n")
        
        if num_successful > 0:
            f.write("### ⏱️ Total and Average Metrics\n")
            f.write(f"  Total Dataset Duration: **{total_duration_formatted}**\n")
            f.write(f"  Total Dataset Size: **{round(total_size_mb / 1024, 2)} GB** ({round(total_size_mb, 2)} MB)\n")
            f.write(f"  Average Video Duration: {str(timedelta(seconds=round(total_duration_s / num_successful)))}\n")
            f.write(f"  Average Video Size: {round(total_size_mb / num_successful, 2)} MB\n\n")
            
            f.write("### 🖼️ Resolution Distribution\n")
            for res, count in sorted(resolution_counts.items(), key=lambda item: item[1], reverse=True):
                f.write(f"  {res}: {count} videos ({round((count/num_successful)*100, 1)}%)\n")
            f.write("\n")
            
            if num_errors > 0:
                f.write("### ⚠️ Files with Processing Errors\n")
                for s in stats_list:
                    if 'Error' in s:
                        f.write(f"  {s['Path']} -> {s['Error']}\n")
        else:
             f.write("No videos were successfully processed.\n")


def run_stats_extraction():
    """Traverses the directory, extracts stats, and logs the summary."""
    all_video_stats = []
    
    print(f"Starting to scan directory: {ROOT_DIR}")
    
    # Traverse the directory
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(root, file)
                # Instead of printing every file, let's just show progress
                # print(f"  Processing: {video_path}") 
                stats = get_video_stats(video_path)
                all_video_stats.append(stats)
                
    # Log the summary statistics
    log_summary_to_file(all_video_stats, OUTPUT_LOG_FILE)
    
    print("-" * 40)
    print(f"✅ Summary generated for {len(all_video_stats)} total files.")
    print(f"Results saved to: {os.path.abspath(OUTPUT_LOG_FILE)}")

if __name__ == "__main__":
    # Make sure you have 'moviepy' installed: pip install moviepy
    run_stats_extraction()