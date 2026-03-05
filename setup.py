from setuptools import setup, find_packages

setup(
    name="bhaskera",
    version="0.3.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
