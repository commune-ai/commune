import commune as c
import streamlit as st
import logging
import cad_prompts
import speech_to_text
from openai_text import generate_ai_text
from modeling import text_to_cad, check_model_generation_status
from slice_print import slice_with_prusaslicer, send_gcode_to_printer, find_latest_stl  # Assuming you have a function for AI text generation
import os

class Almeche(c.Module):
    # Main function

    def text2cad(idea:str =  'a lighter', 
                 path="~",
                 use_ai_for_idea = False,
                 use_speech = False,
                 
                 ):
        api_key = c.module('model.openai').api_keys()[0]
        os.environ["OPENAI_API_KEY"] = api_key
        path = os.path.expanduser(path)  # Update with your path


        # Idea generation or input
        if use_speech and not use_ai_for_idea:
            print("Please speak your idea for a CAD object:")
            idea = speech_to_text.recognize_speech()
        elif use_ai_for_idea:
            print("Generating idea using AI...")
            idea = generate_ai_text(cad_prompts.IDEA_GENERATION, 0.8)  # Adjust temperature as needed

        st.write(f"Your idea is: {idea}, Generating")
        # Generate manufacturing instructions
        instructions_prompt = cad_prompts.MANUFACTURING_INSTRUCTIONS.format(user_idea=idea)
        manufacturing_instructions = generate_ai_text(instructions_prompt, 0.8)
        print(f"Manufacturing Instructions:\n{manufacturing_instructions}")

        # Generate the CAD model using the final instructions
        operation_id = text_to_cad(manufacturing_instructions, "stl")
        
        if operation_id:
            print(f"CAD model generation initiated. Operation ID: {operation_id}")
            
            
            while True:
                result = check_model_generation_status(operation_id)
                if result and result.get("status") == "completed":
                    print("Model generation completed successfully.")
                    stl_path = find_latest_stl(path)
                    slice_with_prusaslicer(stl_path)  # Slicing the STL
                    
                    return {
                        'status': 'success',
                        'path': stl_path
                    }

        
    @classmethod
    def dashboard(cls):
        self = cls()
        import streamlit as st
        c.load_style()
        input_text = st.text_area("Enter text here", height=100)
        generate_button =  st.button("Generate CAD")
        response = {}
        if generate_button:
            response = self.text2cad(input_text)
        path = response.get('path', None)
        st.write(response)


Almeche.run(__name__)
