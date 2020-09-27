from setuptools import setup, find_packages

setup(
    name="PerceptionAlgo",
    version="0.1",
    author='Technion Formula Team',
    author_email='technionfs@gmail.com',
    url="https://github.com/TechnionAVFormula/Perception_Algo",
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'requests',
        'pandas',
        'torchvision==0.3.0',
        'opencv_python==4.1.0.25',
        'torch==1.1.0',
        'imgaug==0.3.0',
        'onnx==1.6.0',
        'optuna==0.19.0',
        'Pillow==6.2.1',
        'protobuf==3.11.0',
        'pymysql==0.9.3',
        'retrying==1.3.3',
        'tensorboardX==1.9',
        'tqdm==4.39.0',
    ],
)
