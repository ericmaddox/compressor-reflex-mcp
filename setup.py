from setuptools import setup, find_packages

setup(
    name="compressor-reflex-mcp",
    version="0.2.2",
    packages=find_packages(),
    install_requires=[
        "onnxruntime>=1.16.0",
        "transformers>=4.38.0",
        "huggingface-hub>=0.20.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "compressor-reflex-mcp=compressor_reflex_mcp.cli:main",
        ],
    },
)
