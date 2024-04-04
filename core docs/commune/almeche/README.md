# AlmechE: Materializing Ideas

Embrace the future with AlmechE, your personalized AI Engineer! Designed to articulate your concepts into tangible reality, AlmechE streamlines the creative process, requiring only a minimal understanding of the technical aspects on your part. This README aims to facilitate your journey through the set-up process, helping you use AlmechE to embody your ideas into physical objects.

## Overview

AlmechE embodies an amalgamation of technologies, automating the transformation of a mere idea spoken into the system into a 3D printed physical object. The workflow elucidates the process:

1. **Idea Generation**: Converse your idea directly into the system, or capitalize on the AI to generate one for you.
2. **Manufacturing Instructions**: Based on the articulated idea, AlmechE formulates detailed manufacturing instructions.
3. **CAD Model Construction**: The aforementioned instructions are deployed to create a CAD model.
4. **Slicing**: The CAD model undergoes slicing, preparing it explicitly for 3D printing.
5. **Printing**: Finally, the sliced model is projected into the 3D printer, materializing your idea into a physical accessory.

## Pre-requisites

Prior to the commencement, validate the availability of the following:

- **Python**: Python needs to be installed on your system as AlmechE is constructed in the language.
- **Microphone**: A fully functional microphone for effective speech recognition.
- **3D Printer**: A 3D printer that is already connected and configured to your system.

## Installation

### 1. Environment Setup

Install Python via the provided link and ensure it's integrated into your system's PATH.

### 2. AlmechE Download

Obtain AlmechE scripts from the given GitHub repository and store them at an accessible location on your computer.

### 3. Library Installation

Launch your command prompt or terminal, navigate to the AlmechE directory and execute the following commands to embed the essential libraries:

```bash
pip install SpeechRecognition
pip install PyAudio     # Required for microphone input
pip install openai
```

If PyAudio presents any installation obstacles, refer to the comprehensive installation guide provided in the link.

### 4. OpenAI API Key Setup

AlmechE adopts OpenAI's GPT-3 for superior natural language processing, hence, procuring an API key from OpenAI is obligatory.

- Visit the OpenAI website, sign up or log in, as necessary.
- Within your dashboard, traverse to API and adhere to the instructions to get your key.
- Subsequently, set up your key as an environment variable on your system:

    For Windows:
    ```bash
    setx OPENAI_API_KEY "your-api-key-here"
    ```

    For Mac/Linux:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

### 5. 3D Printer Configuration

Ensure your 3D printer is successfully connected, configured, and recognizable by your system. Steps may differ according to the printer model and your operating system.

## Usage

Upon complete setup, AlmechE is all set to serve you:

- Navigate to the AlmechE directory via Terminal or Command Prompt.
- Launch AlmechE using: `python main.py`
- Interact with AlmechE, following the displayed prompts. You can choose to voice out your idea or let the AI generate one for you.

## Troubleshooting

- **Microphone Issues**: Ensure your microphone is set as the default recording device and isn’t muted.
- **PyAudio Installation Issues**: If you face difficulties with PyAudio installation, consider installing it from a wheel file as noted in the guide.
- **3D Printer Connectivity**: Verify if the printer is correctly connected and acknowledged by your system.

## Support

For any queries, issues or support, feel free to refer to the AlmechE Issues page on GitHub.

## Contributing

Your contributions to enhance AlmechE are always welcomed! Please refer to the Contributing Guidelines for a detailed layout.

## License

AlmechE is governed by the MIT License, ensuring open accessibility.
