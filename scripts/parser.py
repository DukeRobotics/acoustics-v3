"""Parse Logic 2 recordings and run analysis on each epoch."""
import os
import csv
import re
from datetime import datetime
from controller import analyze_one_sample


def parse_recordings(paths_to_analyze, output_path="analysis"):
    """Parse all recordings in given paths and write to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    csv_path = os.path.join(output_path, f"analysis_{timestamp}.csv")
    
    # Column structure
    base_fields = ['PATH', 'CLOSEST_HYDROPHONE', 'DISTANCE', 'ALL_VALID', 'PREDICTED', 'IS_NEARBY']
    hydrophones = ['H0', 'H1', 'H2', 'H3']
    h_fields = []
    for h in hydrophones:
        h_fields.extend([
            f'{h} TOA', f'{h} VALID', f'{h} REASON', f'{h} IS_NEARBY', f'{h} CONFIDENCE',
            f'{h} RAW_spectral_flatness', f'{h} FILTERED_spectral_centroid_hz',
            f'{h} FILTERED_time_to_secondary_peak_ms', f'{h} RAW_rise_time_ms'
        ])
   
    fieldnames = base_fields + h_fields
    
    # Write header immediately
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    
    for parent_path in paths_to_analyze:
        if not os.path.exists(parent_path):
            print(f"Path not found: {parent_path}")
            continue
        
        parent_name = os.path.basename(parent_path)
        closest_h = re.search(r'(H\d)', parent_name).group(1) if re.search(r'(H\d)', parent_name) else None
        distance = re.search(r'(\d+ft)', parent_name, re.IGNORECASE).group(1).upper() if re.search(r'(\d+ft)', parent_name, re.IGNORECASE) else None
        
        print(f"\nProcessing: {parent_name}")
        
        for epoch_folder in sorted([d.name for d in os.scandir(parent_path) if d.is_dir()]):
            try:
                is_nearby, is_valid, toa_results, nearby_results = analyze_one_sample(os.path.join(parent_path, epoch_folder))
                
                # Build row
                row = {f: '' for f in fieldnames}
                row.update({
                    'PATH': epoch_folder,
                    'CLOSEST_HYDROPHONE': closest_h or '',
                    'DISTANCE': distance or '',
                    'PREDICTED': 0,
                    'IS_NEARBY': is_nearby,
                    'ALL_VALID': is_valid,
                })
                
                # Fill hydrophone data from TOA results
                for toa in toa_results:
                    h_idx = toa.get('hydrophone_idx', -1)
                    if 0 <= h_idx < 4:
                        h = f'H{h_idx}'
                        row[f'{h} TOA'] = toa.get('toa_time', '')
                        row[f'{h} VALID'] = toa.get('is_valid', False)
                        row[f'{h} REASON'] = toa.get('validity_reason', '')
                
                # Fill nearby data (with ML model confidence and features)
                for nearby in nearby_results:
                    h_idx = nearby.get('hydrophone_idx', -1)
                    if 0 <= h_idx < 4:
                        h = f'H{h_idx}'
                        row[f'{h} IS_NEARBY'] = nearby.get('is_nearby', False)
                        row[f'{h} CONFIDENCE'] = nearby.get('confidence', '')
                        
                        # Extract feature values
                        feat_vals = nearby.get('feature_values', {})
                        row[f'{h} RAW_spectral_flatness'] = feat_vals.get('RAW_spectral_flatness', '')
                        row[f'{h} FILTERED_spectral_centroid_hz'] = feat_vals.get('FILTERED_spectral_centroid_hz', '')
                        row[f'{h} FILTERED_time_to_secondary_peak_ms'] = feat_vals.get('FILTERED_time_to_secondary_peak_ms', '')
                        row[f'{h} RAW_rise_time_ms'] = feat_vals.get('RAW_rise_time_ms', '')               
                # Append row
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                
                print(f"  {epoch_folder}: OK")
            except Exception as e:
                print(f"  {epoch_folder}: ERROR - {e}")
    
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    paths_to_analyze = [
        "data/4.11.2026/H0_5ft_t1_2026-04-11--15-27-13",
        "data/4.11.2026/H0_10ft_t1_2026-04-11--15-32-45",
        "data/4.11.2026/H0_15ft_t1_2026-04-11--15-38-49",
        "data/4.11.2026/H0_20ft_t1_2026-04-11--15-45-46",
    ]
    
    parse_recordings(paths_to_analyze)
