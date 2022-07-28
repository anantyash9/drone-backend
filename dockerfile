FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7


WORKDIR '/app'
RUN git clone https://github.com/nikhilt1998/yolov5.git  # clone

RUN pip install -qr yolov5/requirements.txt  # install

COPY ./requirements.txt /app/

RUN pip install --upgrade deepforest albumentations==0.5.1 pyyaml
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip uninstall -y opencv-python-headless==4.5.5.62
RUN pip install opencv-python-headless==4.1.2.30

COPY ./checkpoint_20.pl /app/
COPY ./detectionPipeline.py /app/
COPY ./main.py /app/
COPY ./models.py /app/
COPY ./rediscli.py /app/
COPY ./tiles.py /app/
COPY ./utils.py /app/
COPY ./config.py /app/
COPY ./sample.py /app/
COPY ./trainPipeline.py /app/
COPY ./start.sh /app/



RUN chmod +x ./start.sh

CMD ["bash","start.sh"]



