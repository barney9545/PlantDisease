#create a base image
FROM python:3.10-slim

#copy all the files from the current directory to the "app" directory in the container
COPY . /app

#set the working directory to the "app" directory
WORKDIR /app

#run pip install -r requirements.txt for installing the dependencies in docker container
RUN pip install -r requirements.txt

#expose the port 80 for the application to run on HTTP
EXPOSE 80

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit

#run the application
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
