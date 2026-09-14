from setuptools import setup, find_packages

setup(
    name="accident_detection",
    version="0.1.0",
    description="Automated Traffic Accident Detection System using YOLO11, ByteTrack, and Multi-modal Cascade Fusion",
    author="Antigravity AI Team",
    author_email="contact@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "gradio>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)