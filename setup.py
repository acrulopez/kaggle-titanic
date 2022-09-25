from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "joblib==1.1.0",
    "numpy==1.19.5",
    "pandas==0.25.3",
    "patsy==0.5.2",
    "python-dateutil==2.8.2",
    "pytz==2022.2.1",
    "scikit-learn==0.24.2",
    "scipy==1.5.4",
    "six==1.16.0",
    "statsmodels==0.12.2",
    "threadpoolctl==3.1.0",
    "gcsfs==0.7.1",
    "google-cloud-bigquery-storage==1.0.0",
    "google-cloud-storage==2.0.0",
]

setup(
    name="mlops",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),  # Automatically find packages within this directory or below.
    include_package_data=True,  # if packages include any data files, those will be packed together.
    description="Classification training titanic survivors prediction model",
)
