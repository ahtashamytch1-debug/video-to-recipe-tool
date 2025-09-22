FROM python:3.9-slim

Set the working directory in the container
WORKDIR /app

Copy the requirements file into the container
COPY requirements.txt .

Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

Copy the Streamlit app file into the container
COPY streamlit_app.py .

Expose the port for the Streamlit app
EXPOSE 8501

Command to run the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
