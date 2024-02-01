FROM dashboard-commune:latest

# Copy the script into the image
COPY Yolo.sh /usr/local/bin/

# Make the script executable
RUN chmod +x /usr/local/bin/Yolo.sh

RUN pip install ultralytics

# Expose port 7860
EXPOSE 7860

# Run the custom script during the build process
CMD [ "/usr/local/bin/Yolo.sh" ]