from setuptools import find_packages, setup

with open("./requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ecg_bench",
    version="0.0.4",
    packages=find_packages(),
    url="https://github.com/willxxy/ECG-Bench",
    license="MIT",
    author="William Jongwon Han",
    author_email="wjhan@andrew.cmu.edu",
    description="Open source code of ECG-Bench",
    install_requires=required,
)
