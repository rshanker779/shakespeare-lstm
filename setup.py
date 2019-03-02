from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(name="shakespeare-lstm",
      version='1.0.0',
      author="rshanker779",
      author_email="rshanker779@gmail.com",
      description="Letter by letter text generation using an LSTM trained on Shakespeare plays.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/rshanker779/shakespeare-lstm",
      license='MIT',
      python_requires='>=3.5',
      install_requires=[
          'keras',
          'nltk',
          'numpy',
          'tensorflow',
          'setuptools'],
      packages=['src'],

      )
