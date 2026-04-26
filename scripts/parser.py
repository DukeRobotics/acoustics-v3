"""Parse Logic 2 recordings and run analysis on each epoch."""
import os
import csv
import re
from datetime import datetime
from controller import analyze_one_sample


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT = os.path.join(_SCRIPTS_DIR, '..', 'analysis')


def parse_recordings(paths_to_analyze, output_path=_DEFAULT_OUTPUT):
    """Parse all recordings in given paths and write to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"analysis_{timestamp}.csv")
    
    # Column structure
    base_fields = ['PATH', 'CLOSEST_HYDROPHONE', 'DISTANCE', 'ALL_VALID', 'PREDICTED', 'IS_NEARBY']
    hydrophones = ['H0', 'H1', 'H2', 'H3']
    
    # Build TOA/Nearby columns
    h_fields = []
    for h in hydrophones:
        h_fields.extend([f'{h} TOA', f'{h} VALID', f'{h} REASON', f'{h} IS_NEARBY', f'{h} PING_WIDTH'])
    
    # Build feature columns (dynamically discovered from first sample)
    feature_fields = []
    feature_results_first = None
    
    # Discover features from first epoch
    for parent_path in paths_to_analyze:
        if not os.path.exists(parent_path):
            continue
        
        for epoch_folder in sorted([d.name for d in os.scandir(parent_path) if d.is_dir()]):
            try:
                is_nearby, is_valid, toa_results, nearby_results, feature_results = analyze_one_sample(
                    os.path.join(parent_path, epoch_folder)
                )
                if feature_results and len(feature_results) > 0:
                    feature_results_first = feature_results[0]
                    break
            except:
                pass
        if feature_results_first:
            break
    
    # Build feature field names from first result
    if feature_results_first:
        for h_idx in range(4):
            for feature_name in sorted(feature_results_first.keys()):
                if feature_name not in ['hydrophone_idx']:
                    feature_fields.append(f'H{h_idx} {feature_name}')
    
    fieldnames = base_fields + h_fields + feature_fields
    
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
                is_nearby, is_valid, toa_results, nearby_results, feature_results = analyze_one_sample(
                    os.path.join(parent_path, epoch_folder)
                )
                
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
                
                # Fill nearby data
                for nearby in nearby_results:
                    h_idx = nearby.get('hydrophone_idx', -1)
                    if 0 <= h_idx < 4:
                        h = f'H{h_idx}'
                        row[f'{h} IS_NEARBY'] = nearby.get('nearby', False)
                        row[f'{h} PING_WIDTH'] = nearby.get('delta_t', '')
                
                # Fill feature data
                for features in feature_results:
                    h_idx = features.get('hydrophone_idx', -1)
                    if 0 <= h_idx < 4:
                        for feature_name, feature_value in features.items():
                            if feature_name not in ['hydrophone_idx']:
                                col_name = f'H{h_idx} {feature_name}'
                                if col_name in row:
                                    row[col_name] = feature_value
                
                # Append row
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                
                print(f"  {epoch_folder}: OK")
            except Exception as e:
                print(f"  {epoch_folder}: ERROR - {e}")
    
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    paths_to_analyze = [
        "C:\\Users\\suvas\\Documents\\acoustics-v3\\data\\2.22.2026\\H0_Closest_0FT_2026-02-22--15-29-08",
        "C:\\Users\\suvas\\Documents\\acoustics-v3\\data\\2.22.2026\\H0_Closest_10FT_2026-02-22--15-35-26",
        "C:\\Users\\suvas\\Documents\\acoustics-v3\\data\\2.22.2026\\H0_Closest_20FT_2026-02-22--15-41-55",
        "C:\\Users\\suvas\\Documents\\acoustics-v3\\data\\2.22.2026\\H0_Closest_30FT_2026-02-22--15-47-28",
        "C:\\Users\\suvas\\Documents\\acoustics-v3\\data\\2.22.2026\\H0_Closest_40FT_2026-02-22--15-53-23",
    ]
    
    parse_recordings(paths_to_analyze)
