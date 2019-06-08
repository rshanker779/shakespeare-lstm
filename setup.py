import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="shakespeare-lstm",
    version="1.0.0",
    author="rshanker779",
    author_email="rshanker779@gmail.com",
    description="Letter by letter text generation using an LSTM trained on Shakespeare plays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rshanker779/shakespeare-lstm",
    license="MIT",
    python_requires=">=3.5",
    install_requires=["keras", "nltk", "numpy", "setuptools", "setuptools-git"],
    packages=find_packages(),
    entry_points={"console_scripts": ["shake_lstm=src.lstm_shakespeare:main"]},
    extras_require={"tf": ["tensorflow>=1.0.0"], "tf_gpu": ["tensorflow-gpu>=1.0.0"]},
    include_package_data=True,
)
