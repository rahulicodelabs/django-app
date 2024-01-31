# base image
FROM python:3.10

# Install Tesseract and other dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Ghostscript and other dependencies
RUN apt-get update && apt-get install -y \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/www/html/PROD/pyapps

# copy the requirement file into your directory requirement file.
COPY requirements.txt .

# run this command to install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK and download stopwords and wordnet
RUN python -m nltk.downloader stopwords wordnet

# copy whole project to your docker home directory.
COPY . .

# port where the Django app runs
EXPOSE 8000

# start server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
