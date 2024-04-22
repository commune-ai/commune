import commune as c
import gradio as gr
import torch
import numpy as np
import requests
import random
from io import BytesIO
from utils import *
from constants import *
from pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from torch import autocast, inference_mode
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from transformers import AutoProcessor, BlipForConditionalGeneration
from share_btn import community_icon_html, loading_icon_html, share_js

# load pipelines
# sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_model_id = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = SemanticStableDiffusionImg2ImgPipeline_DPMSolver.from_pretrained(sd_model_id,vae=vae,torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(sd_model_id, subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)

blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

## IMAGE CPATIONING ##
def caption_image(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption, generated_caption

def sample(zs, wts, attention_store, text_cross_attention_maps, prompt_tar="", cfg_scale_tar=15, skip=36, eta=1):
    latents = wts[-1].expand(1, -1, -1, -1)
    img, attention_store, text_cross_attention_maps = pipe(
        prompt=prompt_tar,
        init_latents=latents,
        guidance_scale=cfg_scale_tar,
        # num_images_per_prompt=1,
        # num_inference_steps=steps,
        # use_ddpm=True,
        # wts=wts.value,
        attention_store = attention_store, text_cross_attention_maps=text_cross_attention_maps,
        zs=zs,
    )
    return img.images[0], attention_store, text_cross_attention_maps


def reconstruct(
    tar_prompt,
    image_caption,
    tar_cfg_scale,
    skip,
    wts,
    zs,
    attention_store,
    text_cross_attention_maps,
    do_reconstruction,
    reconstruction,
    reconstruct_button,
):
    if reconstruct_button == "Hide Reconstruction":
        return (
            reconstruction,
            reconstruction,
            gr.update(visible=False),
            do_reconstruction,
            "Show Reconstruction",
        )

    else:
        if do_reconstruction:
            if (
                image_caption.lower() == tar_prompt.lower()
            ):  # if image caption was not changed, run actual reconstruction
                tar_prompt = ""
            latents = wts[-1].expand(1, -1, -1, -1)
            reconstruction, attention_store, text_cross_attention_maps = sample(
                zs, wts, attention_store=attention_store, text_cross_attention_maps=text_cross_attention_maps,prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale
            )
            do_reconstruction = False
        return (
            reconstruction,
            reconstruction,
            gr.update(visible=True),
            do_reconstruction,
            "Hide Reconstruction",
        )


def load_and_invert(
    input_image,
    do_inversion,
    seed,
    randomize_seed,
    wts,
    zs,
    src_prompt="",
    # tar_prompt="",
    steps=30,
    src_cfg_scale=3.5,
    skip=15,
    tar_cfg_scale=15,
    progress=gr.Progress(track_tqdm=True),
):
    # x0 = load_512(input_image, device=device).to(torch.float16)

    if do_inversion or randomize_seed:
        seed = randomize_seed_fn(seed, randomize_seed)
        seed_everything(seed)
        # invert and retrieve noise maps and latent
        zs_tensor, wts_tensor = pipe.invert(
            image_path=input_image,
            source_prompt=src_prompt,
            source_guidance_scale=src_cfg_scale,
            num_inversion_steps=steps,
            skip=skip,
            eta=1.0,
        )
        wts = wts_tensor
        zs = zs_tensor
        do_inversion = False

    return wts, zs, do_inversion, gr.update(visible=False)

## SEGA ##

def edit(input_image,
            wts, zs, attention_store, text_cross_attention_maps,
            tar_prompt,
            image_caption,
            steps,
            skip,
            tar_cfg_scale,
            edit_concept_1,edit_concept_2,edit_concept_3,
            guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
            warmup_1, warmup_2, warmup_3,
            neg_guidance_1, neg_guidance_2, neg_guidance_3,
            threshold_1, threshold_2, threshold_3,
            do_reconstruction,
            reconstruction,  
            # for inversion in case it needs to be re computed (and avoid delay):
            do_inversion,
            seed, 
            randomize_seed,
            src_prompt,
            src_cfg_scale,
            mask_type,
            progress=gr.Progress(track_tqdm=True)):
    show_share_button = gr.update(visible=True)
    if(mask_type == "No mask"):
        use_cross_attn_mask = False
        use_intersect_mask = False
    elif(mask_type=="Cross Attention Mask"):
        use_cross_attn_mask = True
        use_intersect_mask = False 
    elif(mask_type=="Intersect Mask"):
        use_cross_attn_mask = False
        use_intersect_mask = True 

    if randomize_seed:
        seed = randomize_seed_fn(seed, randomize_seed)
    seed_everything(seed)

    if do_inversion or randomize_seed:
        zs_tensor, wts_tensor = pipe.invert(
           image_path = input_image,
           source_prompt =src_prompt,
           source_guidance_scale= src_cfg_scale,
           num_inversion_steps = steps,
           skip = skip,
           eta = 1.0,
           )
        wts = wts_tensor
        zs = zs_tensor
        do_inversion = False
    
    if image_caption.lower() == tar_prompt.lower(): # if image caption was not changed, run pure sega
          tar_prompt = ""
        
    if edit_concept_1 != "" or edit_concept_2 != "" or edit_concept_3 != "":
      editing_args = dict(
      editing_prompt = [edit_concept_1,edit_concept_2,edit_concept_3],
      reverse_editing_direction = [ neg_guidance_1, neg_guidance_2, neg_guidance_3,],
      edit_warmup_steps=[warmup_1, warmup_2, warmup_3,],
      edit_guidance_scale=[guidnace_scale_1,guidnace_scale_2,guidnace_scale_3],
      edit_threshold=[threshold_1, threshold_2, threshold_3],
      edit_momentum_scale=0,
      edit_mom_beta=0,
      eta=1,
      use_cross_attn_mask=use_cross_attn_mask,
      use_intersect_mask=use_intersect_mask
      )

      latnets = wts[-1].expand(1, -1, -1, -1)
      sega_out, attention_store, text_cross_attention_maps = pipe(prompt=tar_prompt, 
                          init_latents=latnets, 
                          guidance_scale = tar_cfg_scale,
                          # num_images_per_prompt=1,
                          # num_inference_steps=steps,
                          # use_ddpm=True,  
                          # wts=wts.value, 
                          zs=zs, attention_store=attention_store, text_cross_attention_maps=text_cross_attention_maps, **editing_args)
      
      return sega_out.images[0], gr.update(visible=True), do_reconstruction, reconstruction, wts, zs, attention_store, text_cross_attention_maps, do_inversion, show_share_button
    
    
    else: # if sega concepts were not added, performs regular ddpm sampling
      
      if do_reconstruction: # if ddpm sampling wasn't computed
          pure_ddpm_img, attention_store, text_cross_attention_maps = sample(zs, wts, attention_store=attention_store, text_cross_attention_maps=text_cross_attention_maps, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale)
          reconstruction = pure_ddpm_img
          do_reconstruction = False
          return pure_ddpm_img, gr.update(visible=False), do_reconstruction, reconstruction, wts, zs, attention_store, text_cross_attention_maps, do_inversion, show_share_button
      
      return reconstruction, gr.update(visible=False), do_reconstruction, reconstruction, wts, zs, attention_store, text_cross_attention_maps, do_inversion, show_share_button
        

def randomize_seed_fn(seed, is_random):
    if is_random:
        seed = random.randint(0, np.iinfo(np.int32).max)
    return seed

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def crop_image(image):
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image
    

def get_example():
    case = [
        [
            'examples/car_input.png', 
            # '',
            'cherry blossom', 'green cabriolet','yellow car',
   
             'examples/car_output.png',
            
            
            13,11,7,
            2,2,2,
            False, False, True,
            50,
            25,
            7.5,
            0.65, 0.8, 0.8,
            890000000
           
             ],
        [
            'examples/girl_with_pearl_earring_input.png', 
            # '',
            'glasses', '','',

             'examples/girl_with_pearl_earring_output.png',
            
            
            4,7,0,
            3,2,2,
            False,False,False,
            50,
            25,
            5,
            0.97, 0.95,0.95,
            1900000000
           
             ],
        
                 [
            'examples/flower_field_input.jpg', 
            # '',
            'pink tulips', 'red flowers',
            'van gogh painting',
             'examples/flower_field_output.png',


            20,7,10,
            1,1,1,
                     False,True,False,
                      50,
            25,
            7,
                     0.9, 0.9,0.8,
            1900000000
                     
            
             ],
       
 ]
    return case


def swap_visibilities(input_image,  
                    edit_concept_1,
                    edit_concept_2,
                     edit_concept_3,
                    sega_edited_image,
                    guidnace_scale_1,
                    guidnace_scale_2,
                      guidnace_scale_3,
                    warmup_1,
                    warmup_2,
                      warmup_3,
                    neg_guidance_1,
                    neg_guidance_2,
                      neg_guidance_3,
                    steps,
                    skip,
                    tar_cfg_scale,
                    threshold_1,
                    threshold_2,
                      threshold_3,
                    sega_concepts_counter
                    
):
    sega_concepts_counter=0
    concept1_update = update_display_concept("Remove" if neg_guidance_1 else "Add", edit_concept_1, neg_guidance_1, sega_concepts_counter)
    if(edit_concept_2 != ""):
        concept2_update = update_display_concept("Remove" if neg_guidance_2 else "Add", edit_concept_2, neg_guidance_2, sega_concepts_counter+1)
    else:
        concept2_update = gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(value=neg_guidance_2),gr.update(visible=True),gr.update(visible=False),sega_concepts_counter+1
    
    return (gr.update(visible=True), *concept1_update[:-1], *concept2_update)
    


########
# demo #
########


intro = """
<div style="display: flex;align-items: center;justify-content: center">
    <img src="https://huggingface.co/spaces/editing-images/leditsplusplus/resolve/main/magician.png" width="100" style="display: inline-block">
    <h1 style="margin-left: 12px;text-align: center;margin-bottom: 7px;display: inline-block">LEDITS++</h1>
    <h3 style="display: inline-block;margin-left: 10px;margin-top: 6px;font-weight: 500">Limitless Image Editing using Text-to-Image Models</h3>
</div>

<p style="font-size: 0.95rem;margin: 0rem;line-height: 1.2em;margin-top:1em;display: inline-block">
    <a href="https://leditsplusplus-project.static.hf.space" target="_blank">project page</a> | <a href="https://arxiv.org/abs/2311.16711" target="_blank">paper</a>
     | 
    <a href="https://huggingface.co/spaces/leditsplusplus/demo?duplicate=true" target="_blank" style="
        display: inline-block;
    ">
    <img style="margin-top: -1em;margin-bottom: 0em;position: absolute;" src="https://bit.ly/3CWLGkA" alt="Duplicate Space"></a>
</p>
"""


with gr.Blocks(css="style.css") as demo:
    def update_counter(sega_concepts_counter, concept1, concept2, concept3):
        if sega_concepts_counter == "":
            sega_concepts_counter = sum(1 for concept in (concept1, concept2, concept3) if concept != '')
        return sega_concepts_counter
    def remove_concept(sega_concepts_counter, row_triggered):
      sega_concepts_counter -= 1
      rows_visibility = [gr.update(visible=False) for _ in range(4)]
      
      if(row_triggered-1 > sega_concepts_counter):
            rows_visibility[sega_concepts_counter] = gr.update(visible=True)
      else:
            rows_visibility[row_triggered-1] = gr.update(visible=True)
      
      row1_visibility, row2_visibility, row3_visibility, row4_visibility = rows_visibility

      guidance_scale_label = "Concept Guidance Scale"
      # enable_interactive =  gr.update(interactive=True)
      return (gr.update(visible=False),
              gr.update(visible=False, value="",),
              gr.update(interactive=True, value=""),
              gr.update(visible=False,label = guidance_scale_label),
              gr.update(interactive=True, value =False),
              gr.update(value=DEFAULT_WARMUP_STEPS),
              gr.update(value=DEFAULT_THRESHOLD),
              gr.update(visible=True),
              gr.update(interactive=True, value="custom"),
              row1_visibility,
              row2_visibility,
              row3_visibility,
              row4_visibility,
              sega_concepts_counter
             ) 
    
    
    
    def update_display_concept(button_label, edit_concept, neg_guidance, sega_concepts_counter):
      sega_concepts_counter += 1
      guidance_scale_label = "Concept Guidance Scale"
      if(button_label=='Remove'):
        neg_guidance = True
        guidance_scale_label = "Negative Guidance Scale" 
      
      return (gr.update(visible=True), #boxn
             gr.update(visible=True, value=edit_concept), #concept_n
             gr.update(visible=True,label = guidance_scale_label), #guidance_scale_n
             gr.update(value=neg_guidance),#neg_guidance_n
             gr.update(visible=False), #row_n
             gr.update(visible=True), #row_n+1
             sega_concepts_counter
             ) 


    def display_editing_options(run_button, clear_button, sega_tab):
      return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    
    def update_interactive_mode(add_button_label):
      if add_button_label == "Clear":
        return gr.update(interactive=False), gr.update(interactive=False)
      else:
        return gr.update(interactive=True), gr.update(interactive=True)
    
    def update_dropdown_parms(dropdown):
        if dropdown == 'custom':
          return DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD
        elif dropdown =='style':
          return STYLE_SEGA_CONCEPT_GUIDANCE_SCALE,STYLE_WARMUP_STEPS, STYLE_THRESHOLD
        elif dropdown =='object':
          return OBJECT_SEGA_CONCEPT_GUIDANCE_SCALE,OBJECT_WARMUP_STEPS, OBJECT_THRESHOLD
        elif dropdown =='faces':
          return FACE_SEGA_CONCEPT_GUIDANCE_SCALE,FACE_WARMUP_STEPS, FACE_THRESHOLD


    def reset_do_inversion():
        return True

    def reset_do_reconstruction():
      do_reconstruction = True
      return  do_reconstruction

    def reset_image_caption():
        return ""

    def update_inversion_progress_visibility(input_image, do_inversion):
      if do_inversion and not input_image is None:
          return gr.update(visible=True)
      else:
        return gr.update(visible=False)

    def update_edit_progress_visibility(input_image, do_inversion):
      # if do_inversion and not input_image is None:
      #     return inversion_progress.update(visible=True)
      # else:
        return gr.update(visible=True)


    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    attention_store=gr.State()
    text_cross_attention_maps = gr.State()
    reconstruction = gr.State()
    do_inversion = gr.State(value=True)
    do_reconstruction = gr.State(value=True)
    sega_concepts_counter = gr.State(0)
    image_caption = gr.State(value="")

    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True, elem_id="input_image")
        ddpm_edited_image = gr.Image(label=f"Pure DDPM Inversion Image", interactive=False, visible=False)
        sega_edited_image = gr.Image(label=f"LEDITS Edited Image", interactive=False, elem_id="output_image")

    with gr.Group(visible=False, elem_id="share-btn-wrapper") as share_btn_container:
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=True)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
        
    with gr.Row():
      with gr.Group(visible=False, elem_id="box1") as box1:
        with gr.Row():
          concept_1 = gr.Button(scale=3, value="")
          remove_concept1 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
            guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                            info="How strongly the concept should modify the image",
                                                  value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                  step=0.5, interactive=True)
      with gr.Group(visible=False, elem_id="box2") as box2:
        with gr.Row():
          concept_2 = gr.Button(scale=3, value="")
          remove_concept2 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
          guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                              info="How strongly the concept should modify the image",
                                                    value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                    step=0.5, interactive=True)
      with gr.Group(visible=False, elem_id="box3") as box3:
        with gr.Row():
          concept_3 = gr.Button(scale=3, value="")
          remove_concept3 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
          guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                              info="How strongly the concept should modify the image",
                                                    value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                    step=0.5, interactive=True)


    with gr.Row():
        inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
        
    with gr.Group():
        intro_segs = gr.Markdown("Add/Remove Concepts from your Image <span style=\"font-size: 12px; color: rgb(156, 163, 175)\">with Semantic Guidance</span>")
                  # 1st SEGA concept
        with gr.Row() as row1:
              with gr.Column(scale=3, min_width=100):
                  with gr.Row():
                      # with gr.Column(scale=3, min_width=100):
                            edit_concept_1 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1, value="",
                                              placeholder="E.g.: Sunglasses",
                                          )
                      # with gr.Column(scale=2, min_width=100):# better mobile ui
                            dropdown1 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
    

              with gr.Column(scale=1, min_width=100, visible=False):
                      neg_guidance_1 = gr.Checkbox(
                          label='Remove Concept?')
              
              with gr.Column(scale=1, min_width=100):
                   with gr.Row(): # better mobile ui
                       with gr.Column():
                          add_1 = gr.Button('Add')
                          remove_1 = gr.Button('Remove')
             
    
                  # 2nd SEGA concept
        with gr.Row(visible=False) as row2:
            with gr.Column(scale=3, min_width=100):
                with gr.Row(): #better mobile UI
                    # with gr.Column(scale=3, min_width=100):
                            edit_concept_2 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1,
                                              placeholder="E.g.: Realistic",
                                          )
                    # with gr.Column(scale=2, min_width=100):# better mobile ui
                            dropdown2 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])

            with gr.Column(scale=1, min_width=100, visible=False):
                      neg_guidance_2 = gr.Checkbox(
                          label='Remove Concept?')
                
            with gr.Column(scale=1, min_width=100):
                with gr.Row(): # better mobile ui
                    with gr.Column():
                      add_2 = gr.Button('Add')
                      remove_2 = gr.Button('Remove')
    
                  # 3rd SEGA concept
        with gr.Row(visible=False) as row3:
            with gr.Column(scale=3, min_width=100):
                with gr.Row(): #better mobile UI  
                    # with gr.Column(scale=3, min_width=100):
                            edit_concept_3 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1,
                                              placeholder="E.g.: orange",
                                          )
                    # with gr.Column(scale=2, min_width=100):
                            dropdown3 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
            
            with gr.Column(scale=1, min_width=100, visible=False):
                             neg_guidance_3 = gr.Checkbox(
                              label='Remove Concept?',visible=True)
            
            with gr.Column(scale=1, min_width=100):
                with gr.Row(): # better mobile ui
                    with gr.Column():
                         add_3 = gr.Button('Add')
                         remove_3 = gr.Button('Remove')
    
        with gr.Row(visible=False) as row4:
            gr.Markdown("### Max of 3 concepts reached. Remove a concept to add more")
    
        #with gr.Row(visible=False).style(mobile_collapse=False, equal_height=True):
        #            add_concept_button = gr.Button("+1 concept")


    
    
                # caption_button = gr.Button("Caption Image", scale=1)
        
    
    with gr.Row():
        run_button = gr.Button("Edit your image!", visible=True)
        

    with gr.Accordion("Advanced Options", open=False):
      with gr.Row():
                tar_prompt = gr.Textbox(
                                label="Describe your edited image (optional)",
                                elem_id="target_prompt",
                                # show_label=False,
                                max_lines=1, value="", scale=3,
                                placeholder="Target prompt, DDPM Inversion", info = "DPM Solver++ Inversion Prompt. Can help with global changes, modify to what you would like to see"
                            )
      with gr.Tabs() as tabs:

          with gr.TabItem('General options', id=2):
            with gr.Row():
                with gr.Column(min_width=100):
                   clear_button = gr.Button("Clear", visible=True)
                   src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                   steps = gr.Number(value=50, precision=0, label="Num Diffusion Steps", interactive=True)
                   src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                   mask_type = gr.Radio(choices=["No mask", "Cross Attention Mask", "Intersect Mask"], value="Intersect Mask", label="Mask type")

                with gr.Column(min_width=100):
                    reconstruct_button = gr.Button("Show Reconstruction", visible=False)
                    skip = gr.Slider(minimum=0, maximum=95, value=25, step=1, label="Skip Steps", interactive=True, info = "Percentage of skipped denoising steps. Bigger values increase fidelity to input image")
                    tar_cfg_scale = gr.Slider(minimum=1, maximum=30,value=7.5, label=f"Guidance Scale", interactive=True)
                    seed = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)

          with gr.TabItem('SEGA options', id=3) as sega_advanced_tab:
             # 1st SEGA concept
              gr.Markdown("1st concept")
              with gr.Row():
                  warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_1 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, step=0.01, interactive=True, 
                                          info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

              # 2nd SEGA concept
              gr.Markdown("2nd concept")
              with gr.Row() as row2_advanced:
                  warmup_2 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_2 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                          value=DEFAULT_THRESHOLD,
                                          step=0.01, interactive=True,
                                         info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")
              # 3rd SEGA concept
              gr.Markdown("3rd concept")
              with gr.Row() as row3_advanced:
                  warmup_3 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS, step=1,
                                       interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_3 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, step=0.01,
                                          interactive=True,
                                         info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

    # caption_button.click(
    #     fn = caption_image,
    #     inputs = [input_image],
    #     outputs = [tar_prompt]
    # )
    #neg_guidance_1.change(fn = update_label, inputs=[neg_guidance_1], outputs=[add_1])
    #neg_guidance_2.change(fn = update_label, inputs=[neg_guidance_2], outputs=[add_2])
    #neg_guidance_3.change(fn = update_label, inputs=[neg_guidance_3], outputs=[add_3])
    add_1.click(fn=update_counter,
                inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3],
                outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
    add_2.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_2, edit_concept_2, neg_guidance_2, sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3, sega_concepts_counter],queue=False)
    add_3.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3,row3, row4, sega_concepts_counter],queue=False)
    
    remove_1.click(fn = update_display_concept, inputs=[remove_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
    remove_2.click(fn = update_display_concept, inputs=[remove_2, edit_concept_2, neg_guidance_2 ,sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3,sega_concepts_counter],queue=False)
    remove_3.click(fn = update_display_concept, inputs=[remove_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3, row3, row4, sega_concepts_counter],queue=False)
    
    remove_concept1.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
        fn = remove_concept, inputs=[sega_concepts_counter,gr.State(1)], outputs= [box1, concept_1, edit_concept_1, guidnace_scale_1,neg_guidance_1,warmup_1, threshold_1, add_1, dropdown1, row1, row2, row3, row4, sega_concepts_counter],queue=False)
    remove_concept2.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
        fn = remove_concept,  inputs=[sega_concepts_counter,gr.State(2)], outputs=[box2, concept_2, edit_concept_2, guidnace_scale_2,neg_guidance_2, warmup_2, threshold_2, add_2 , dropdown2, row1, row2, row3, row4, sega_concepts_counter],queue=False)
    remove_concept3.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
        fn = remove_concept,inputs=[sega_concepts_counter,gr.State(3)], outputs=[box3, concept_3, edit_concept_3, guidnace_scale_3,neg_guidance_3,warmup_3, threshold_3,  add_3, dropdown3, row1, row2, row3, row4, sega_concepts_counter],queue=False)

    #add_concept_button.click(fn = update_display_concept, inputs=sega_concepts_counter,
    #           outputs= [row2, row2_advanced, row3, row3_advanced, add_concept_button, sega_concepts_counter], queue = False)

    run_button.click(
        fn=edit,
        inputs=[input_image,
                wts, zs, attention_store,
                text_cross_attention_maps,
                tar_prompt,
                image_caption,
                steps,
                skip,
                tar_cfg_scale,
                edit_concept_1,edit_concept_2,edit_concept_3,
                guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
                warmup_1, warmup_2, warmup_3,
                neg_guidance_1, neg_guidance_2, neg_guidance_3,
                threshold_1, threshold_2, threshold_3, do_reconstruction, reconstruction,
                do_inversion,
                seed, 
                randomize_seed,
                src_prompt,
                src_cfg_scale,
                mask_type
        ],
        outputs=[sega_edited_image, reconstruct_button, do_reconstruction, reconstruction, wts, zs,attention_store, text_cross_attention_maps, do_inversion, share_btn_container]
        
    )
    # .success(fn=update_gallery_display, inputs= [prev_output_image, sega_edited_image], outputs = [gallery, gallery, prev_output_image])


    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue=False,
        concurrency_limit=None
    ).then(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed],
        queue=False,
        concurrency_limit=None
    )
    
    # Automatically start inverting upon input_image change
    input_image.upload(
        fn = crop_image,
        inputs = [input_image],
        outputs = [input_image],
        queue=False,
        concurrency_limit=None,
    ).then(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue=False,
        concurrency_limit=None        
    ).then(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed], 
        queue=False,
        concurrency_limit=None        
    ).then(fn = caption_image,
        inputs = [input_image],
        outputs = [tar_prompt, image_caption],
        queue=False,
        concurrency_limit=None        
    )

    # Repeat inversion (and reconstruction) when these params are changed:
    src_prompt.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False
    ).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    steps.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False
    ).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    src_cfg_scale.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False
    ).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    # Repeat only reconstruction these params are changed:
    tar_prompt.change(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    tar_cfg_scale.change(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    skip.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False
    ).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False
    )

    seed.change(
        fn=reset_do_inversion,
        outputs=[do_inversion],
        queue=False
    ).then(
        fn=reset_do_reconstruction,
        outputs=[do_reconstruction],
        queue=False
    )

    dropdown1.change(fn=update_dropdown_parms, inputs = [dropdown1], outputs = [guidnace_scale_1,warmup_1,  threshold_1], queue=False)
    dropdown2.change(fn=update_dropdown_parms, inputs = [dropdown2], outputs = [guidnace_scale_2,warmup_2,  threshold_2], queue=False)
    dropdown3.change(fn=update_dropdown_parms, inputs = [dropdown3], outputs = [guidnace_scale_3,warmup_3,  threshold_3], queue=False)

    clear_components = [input_image,ddpm_edited_image,ddpm_edited_image,sega_edited_image, do_inversion,
                                   src_prompt, steps, src_cfg_scale, seed,
                                  tar_prompt, skip, tar_cfg_scale, reconstruct_button,reconstruct_button,
                                  edit_concept_1, guidnace_scale_1,guidnace_scale_1,warmup_1,  threshold_1, neg_guidance_1,dropdown1, concept_1, concept_1, row1,
                                  edit_concept_2, guidnace_scale_2,guidnace_scale_2,warmup_2,  threshold_2, neg_guidance_2,dropdown2, concept_2, concept_2, row2,
                                  edit_concept_3, guidnace_scale_3,guidnace_scale_3,warmup_3,  threshold_3, neg_guidance_3,dropdown3, concept_3,concept_3, row3,
                                  row4,sega_concepts_counter, box1, box2, box3 ]

    clear_components_output_vals = [None, None,gr.update(visible=False), None, True,
                     "", DEFAULT_DIFFUSION_STEPS, DEFAULT_SOURCE_GUIDANCE_SCALE, DEFAULT_SEED,
                     "", DEFAULT_SKIP_STEPS, DEFAULT_TARGET_GUIDANCE_SCALE, gr.update(value="Show Reconstruction"),gr.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","", gr.update(visible=False), gr.update(visible=True),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","", gr.update(visible=False), gr.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","",gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=0),
                          gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]


    clear_button.click(lambda: clear_components_output_vals, outputs = clear_components)

    reconstruct_button.click(lambda: ddpm_edited_image.update(visible=True), outputs=[ddpm_edited_image]).then(fn = reconstruct,
                inputs = [tar_prompt,
                image_caption,
                tar_cfg_scale,
                skip,
                wts, zs,
                do_reconstruction,
                reconstruction,
                          reconstruct_button],
                outputs = [ddpm_edited_image,reconstruction, ddpm_edited_image, do_reconstruction, reconstruct_button])

    randomize_seed.change(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed],
        queue = False)

    share_button.click(None, [], [], js=share_js)
    
    gr.Examples(
        label='Examples',
        fn=swap_visibilities,
        run_on_click=True,
        examples=get_example(),
        inputs=[input_image,
                    edit_concept_1,
                    edit_concept_2,
                edit_concept_3,
                    sega_edited_image,
                    guidnace_scale_1,
                    guidnace_scale_2,
                guidnace_scale_3,
                    warmup_1,
                    warmup_2,
                warmup_3,
                    neg_guidance_1,
                    neg_guidance_2,
                neg_guidance_3,
                    steps,
                    skip,
                    tar_cfg_scale,
                    threshold_1, 
                    threshold_2,
                threshold_3,
                    seed,
                    sega_concepts_counter
               ],
        outputs=[share_btn_container, box1, concept_1, guidnace_scale_1,neg_guidance_1, row1, row2,box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3,sega_concepts_counter],
        cache_examples=True
    )


class ModelImageedit(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def gradio() -> int:
        demo.queue(default_concurrency_limit=1)
        demo.launch()