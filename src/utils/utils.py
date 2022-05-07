import base64
import yaml


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./"+fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
        
    return content