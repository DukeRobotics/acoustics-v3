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
    
    # Define the 7 optimal features used by the ML model
    OPTIMAL_FEATURES = [
        'H0_RAW_spectral_flatness',
        'H0_FILTERED_spectral_centroid_hz',
        'H0_FILTERED_time_to_secondary_peak_ms',
        'H0_RAW_rise_time_ms',
        'H0_RAW_spectral_centroid_hz',
        'H0_FILTERED_rise_time_ms',
        'H0_FILTERED_fwhm_ms',
    ]
    
    # Column structure
    base_fields = ['PATH', 'CLOSEST_HYDROPHONE', 'DISTANCE', 'ALL_VALID', 'PREDICTED', 'IS_NEARBY']
    hydrophones = ['H0', 'H1', 'H2', 'H3']
    h_fields = []
    for h in hydrophones:
        h_fields.extend([f'{h} TOA', f'{h} VALID', f'{h} REASON', f'{h} IS_NEARBY', f'{h} CONFIDENCE'])
        # Add optimal features columns (strip H0_ prefix for non-H0 hydrophones)
        for feat in OPTIMAL_FEATURES:
            if h == 'H0':
                h_fields.append(f'{h} {feat}')
            else:
                clean_feat = feat.replace('H0_', '')
                h_fields.append(f'{h} {clean_feat}')
    
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
        distance = re.search(r'(\d+FT)', parent_name).group(1) if re.search(r'(\d+FT)', parent_name) else None
        
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
                        row[f'{h} IS_NEARBY'] = nearby.get('nearby', False)
                        row[f'{h} CONFIDENCE'] = nearby.get('confidence', '')
                        
                        # Extract optimal features
                        optimal_features_dict = nearby.get('optimal_features_values', {})
                        for feat in OPTIMAL_FEATURES:
                            feat_value = optimal_features_dict.get(feat, '')
                            if h == 'H0':
                                row[f'{h} {feat}'] = feat_value
                            else:
                                clean_feat = feat.replace('H0_', '')
                                row[f'{h} {clean_feat}'] = feat_value
                
                # Append row
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                
                print(f"  {epoch_folder}: OK")
            except Exception as e:
                print(f"  {epoch_folder}: ERROR - {e}")
    
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    paths_to_analyze = [
        "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08",
        "data/2.22.2026/H0_Closest_10FT_2026-02-22--15-35-26",
        "data/2.22.2026/H0_Closest_20FT_2026-02-22--15-41-55",
        "data/2.22.2026/H0_Closest_30FT_2026-02-22--15-47-28",
        "data/2.22.2026/H0_Closest_40FT_2026-02-22--15-53-23",
    ]
    
    parse_recordings(paths_to_analyze)
