FROM dashboard-commune:latest

# Copy the script into the image
COPY run-langchain-agent-commune.sh /usr/local/bin/

# Make the script executable
RUN chmod +x /usr/local/bin/run-langchain-agent-commune.sh

ENV OPENAI_API_KEY=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

#RUN  install portaudio19-dev
# Install Python packages
RUN pip install beautifulsoup4 \
    && pip install load_dotenv \
    && pip install duckduckgo-search \
    && pip install SpeechRecognition \
#    && pip install PyAudio \
    && pip3 install langchain_community \
    && pip3 install langchain_openai \
    && pip install openai \
    && pip install pyChatGPT \
    && pip install promptml \
    && pip install langchain


# Install langchain using conda (adjust as needed)
#RUN conda install -c conda-forge langchain

# Expose port 7861
EXPOSE 7861

# Run the custom script during the build process
CMD [ "/usr/local/bin/run-langchain-agent-commune.sh" ]
