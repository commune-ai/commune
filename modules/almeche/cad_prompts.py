# cad_prompts.py

# Prompt for generating innovative product ideas
IDEA_GENERATION = """
Generate simple-to-manufacture yet highly innovative and uniquely designed CAD object ideas.
Think about how to simply address existing real-world problems, design something fun or beautiful, or create new opportunities in a unique way.
Provide a concise description of the most promising idea that we can 3D print.
Keep your response under 20 words.
"""

# Prompt for creating manufacturing instructions based on technical specifications
MANUFACTURING_INSTRUCTIONS = """
Based on the idea '{user_idea}', provide structured manufacturing instructions suitable for 3D printing.
Craft a detailed and concise description that includes essential dimensions, materials, and geometric properties.
If the concept is too complex for a single print, do the best possible version of a simplified Minimum Viable Product (MVP).
Aim for a design that uses one object only. Explain the design in unambiguous terms as if to a novice Mechanical Engineer, focusing on 
practicality and manufacturability. Keep your response concise and under 50 words.
"""
