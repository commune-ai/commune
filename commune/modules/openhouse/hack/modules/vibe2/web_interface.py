import gradio as gr
import time
from vibe_generator import VibeGenerator

def create_vibe_interface():
    """Create a Gradio interface for the VibeGenerator."""
    vibe_gen = VibeGenerator()
    available_vibes = vibe_gen.get_available_vibes()
    
    def start_selected_vibe(vibe_name):
        """Start the selected vibe and return status."""
        result = vibe_gen.start_vibe(vibe_name)
        return f"Started '{vibe_name}' vibe! {result['details']['description']}"
    
    def stop_current_vibe():
        """Stop the current vibe and return status."""
        result = vibe_gen.stop_vibe()
        if result['status'] == 'stopped':
            return f"Stopped '{result['previous_vibe']}' vibe."
        else:
            return "No vibe currently running."
    
    def get_vibe_status():
        """Get the current vibe status."""
        current = vibe_gen.get_current_vibe()
        if current:
            return f"Currently vibing: {current['name']} - {current['details']['description']}"
        else:
            return "No vibe currently active. Select a vibe to begin!"
    
    def refresh_status():
        """Refresh the status display."""
        return get_vibe_status()
    
    def get_vibe_details(vibe_name):
        """Get details about the selected vibe."""
        if not vibe_name:
            return "Select a vibe to see details."
            
        details = vibe_gen.get_vibe_details(vibe_name)
        return f"""
## {vibe_name.title()} Vibe

**Description:** {details['description']}

**Keywords:** {', '.join(details['keywords'])}

**Tempo:** {details['tempo']}

**Intensity:** {details['intensity']:.1f}/1.0

**Color Palette:** {', '.join(details['colors'])}
"""
    
    def create_new_vibe(name, colors, tempo, intensity, keywords, description):
        """Create a new custom vibe."""
        if not name or not description:
            return "Error: Name and description are required.", gr.Dropdown(choices=available_vibes)
            
        try:
            color_list = [c.strip() for c in colors.split(',') if c.strip()]
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            intensity_val = float(intensity)
            
            vibe_gen.create_custom_vibe(
                name=name,
                colors=color_list if color_list else ["#3A86FF", "#8338EC"],
                tempo=tempo,
                intensity=intensity_val,
                keywords=keyword_list if keyword_list else ["custom", "vibe"],
                description=description
            )
            
            # Update available vibes
            new_vibes = vibe_gen.get_available_vibes()
            return f"Created new vibe: {name}!", gr.Dropdown(choices=new_vibes, value=name)
        except Exception as e:
            return f"Error creating vibe: {str(e)}", gr.Dropdown(choices=available_vibes)
    
    # Create the interface
    with gr.Blocks(title="Dope Vibe Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🌈✨ Dope Vibe Generator ✨🌈")
        gr.Markdown("Create and experience different vibes to set your mood just right.")
        
        with gr.Tab("Experience Vibes"):
            status = gr.Markdown(get_vibe_status())
            
            with gr.Row():
                vibe_dropdown = gr.Dropdown(choices=available_vibes, label="Select a Vibe")
                refresh_btn = gr.Button("Refresh Status")
            
            with gr.Row():
                start_btn = gr.Button("Start Vibe", variant="primary")
                stop_btn = gr.Button("Stop Vibe", variant="stop")
            
            vibe_details = gr.Markdown("Select a vibe to see details.")
            
            # Connect the components
            vibe_dropdown.change(get_vibe_details, inputs=vibe_dropdown, outputs=vibe_details)
            start_btn.click(start_selected_vibe, inputs=vibe_dropdown, outputs=status)
            stop_btn.click(stop_current_vibe, outputs=status)
            refresh_btn.click(refresh_status, outputs=status)
        
        with gr.Tab("Create New Vibe"):
            gr.Markdown("## Create Your Custom Vibe")
            
            name_input = gr.Textbox(label="Vibe Name")
            desc_input = gr.Textbox(label="Description", lines=2)
            colors_input = gr.Textbox(label="Colors (comma separated hex codes)", 
                                    placeholder="#FF5733, #33FF57, #3357FF")
            
            with gr.Row():
                tempo_input = gr.Dropdown(
                    choices=["very slow", "slow", "medium", "fast", "very fast"],
                    value="medium",
                    label="Tempo"
                )
                intensity_input = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                    label="Intensity"
                )
            
            keywords_input = gr.Textbox(
                label="Keywords (comma separated)",
                placeholder="chill, relaxed, smooth"
            )
            
            create_btn = gr.Button("Create Vibe", variant="primary")
            create_result = gr.Markdown("")
            
            # Connect create functionality
            create_btn.click(
                create_new_vibe,
                inputs=[
                    name_input, colors_input, tempo_input, 
                    intensity_input, keywords_input, desc_input
                ],
                outputs=[create_result, vibe_dropdown]
            )
    
    return interface

if __name__ == "__main__":
    interface = create_vibe_interface()
    interface.launch()
