import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path
from PIL import Image

def numpy_to_base64(img_array):
    """Convert numpy array to base64 encoded image string."""
    if img_array is None:
        return None
    
    # Ensure uint8 format
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def create_interactive_viewer(data, output_path="src/trajectory_viewer_with_data.html", 
                            template_path="src/trajectory_viewer_template.html", subset_size=None):
    """
    Create an interactive HTML viewer with embedded trajectory data.
    
    :param data: Dataset dictionary with trajectories
    :param output_path: Path to save the HTML file with data
    :param template_path: Path to the HTML template
    :param subset_size: Optional limit on number of trajectories to include
    :returns: Path to the created HTML file
    """
    
    def prepare_json_data(trajectories):
        temp_js_data = []
        for i, traj in enumerate(trajectories):
            if i % 10 == 0:
                print(f"Processing trajectory {i+1}/{len(trajectories)}")
            
            # Convert images to base64
            first_image_data = numpy_to_base64(traj['first_image'])
            last_image_data = numpy_to_base64(traj['last_image'])
            
            # Handle inpainted images (might not exist)
            first_inpainted_data = None
            last_inpainted_data = None
            
            if 'first_image_inpainted' in traj and traj['first_image_inpainted'] is not None:
                first_inpainted_data = numpy_to_base64(traj['first_image_inpainted'])
            
            if 'last_image_inpainted' in traj and traj['last_image_inpainted'] is not None:
                last_inpainted_data = numpy_to_base64(traj['last_image_inpainted'])
            
            # Create trajectory object for JavaScript
            traj_obj = {
                'dataset': traj.get('dataset', 'unknown'),
                'trajectory_id': traj.get('trajectory_id', i),
                'language_instruction': traj.get('language', 'No language instruction available'),
                'first_image_data': first_image_data,
                'last_image_data': last_image_data,
                'first_image_inpainted_data': first_inpainted_data,
                'last_image_inpainted_data': last_inpainted_data
            }
            # Only add target_object if it exists in trajectory data
            if 'language_target_object' in traj:
                traj_obj['target_object'] = traj['language_target_object']

            js_data.append(traj_obj)
        return temp_js_data

    # Load HTML template
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"HTML template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Prepare trajectory data for JavaScript
    trajectories = data['trajectories']
    bridge_trajectories = [traj for traj in trajectories if traj.get('dataset') == 'bridge']
    fractal_trajectories = [traj for traj in trajectories if traj.get('dataset') == 'fractal']

    if subset_size:
        bridge_trajectories = bridge_trajectories[:subset_size // 2]
        fractal_trajectories = fractal_trajectories[:subset_size // 2]

    js_data = []
    print(f"Processing {len(bridge_trajectories)} bridge trajectories...")
    js_data.extend(prepare_json_data(bridge_trajectories))
    print(f"Processing {len(fractal_trajectories)} fractal trajectories...")
    js_data.extend(prepare_json_data(fractal_trajectories))

    # Embed data into HTML
    data_script = f"""
    <script>
        // Trajectory data embedded from Python
        window.trajectoryData = {json.dumps(js_data)};
        
        // Load data when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            if (window.trajectoryData) {{
                loadTrajectoryData(window.trajectoryData);
            }}
        }});
    </script>
    """
    
    # Insert data script before closing </body> tag
    html_content = html_content.replace('</body>', f'{data_script}\n</body>')
    
    # Save the HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\\nInteractive viewer created: {output_path}")
    print(f"Embedded {len(js_data)} trajectories")
    print(f"\\nOpen in browser: file://{output_path.absolute()}")
    
    return str(output_path)