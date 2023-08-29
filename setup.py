import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zarrdataset",
    version=os.environ.get('VERSION', '0.0.0'),
    maintainer="Fernando Cervantes",
    maintainer_email="fernando.cervantes@jax.org",
    description="Zarr-based dataset for PyTorch training pipelines. Written "
                "and maintained by the Research IT team at The Jackson "
                "Laboratory.",

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheJacksonLaboratory/zarrdataset",
    packages=setuptools.find_packages(),
    install_requires=[
        'requests>=2.31.0',
        'aiohttp>=3.8.5',
        'boto-core>=1.31.35',
        'boto3>=1.28.35',
        'fsspec>=2022.11.0',
        's3fs>=0.4.2',
        'zarr>=2.12.0',
        'scikit-image>=0.19.3',
        'dask>=2022.2.0',
        'poisson-disc>=0.2.1',
    ],
    python_requires='>=3.7'
)
