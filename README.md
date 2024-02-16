# Project Setup Guide

## Installation

1. Install Poetry:

   ```
   pip install poetry
   ```
2. Configure Poetry to create virtual environments within the project directory:

   ```
   poetry config virtualenvs.in-project true
   ```
3. Run the setup file:

   ```
   python setup
   ```
4. Install project dependencies using Poetry:

   ```
   poetry install
   ```
## Running the Project

5. Run the project:
   ```
   make run
   ```

### CPU Usage

If CPU usage is sufficient for your needs, the above steps are enough.

### GPU Usage

#### Prerequisites:

- Visual Studio 2022 (latest version) and download the following workloads:
  - Desktop development with C++
  - Universal Windows Platform development
    [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)
- CUDA Toolkit [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

#### Installation Steps:

5. Set environment variable `CMAKE_ARGS` and install the necessary package (For Windows PowerShell):

   ```
   $env:CMAKE_ARGS='-DLLAMA_CUBLAS=on'; poetry run pip install --force-reinstall --no-cache-dir llama-cpp-python
   ```
6. Run the project:

   ```
   make run
   ```
### Additional Note
- For convenience, a pre-configured virtual environment file is available in the release section of the project. If you choose to use this pre-configured virtual environment, please note that you still need to:
  - Have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed for GPU usage.
  - Perform the [first three steps of the installation process](#installation) mentioned above (installing Poetry and configuring it to create virtual environments within the project directory) before activating the provided virtual environment.
