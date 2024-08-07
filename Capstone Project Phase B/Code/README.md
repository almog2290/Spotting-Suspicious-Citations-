# Spotting Suspicious Citations (Code Implementation)

This project aims to identify suspicious citations in academic papers using machine learning models.

## Running the Project on Colab Notebook

You can run the project on Google Colab by clicking the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vKgnY6cUeADSVMuDVkZcsuLRD47gJLwA?usp=sharing)

## Running the Project Locally on a Linux Machine

Follow the instructions below to run the project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/almog2290/Spotting-Suspicious-Citations-.git
    ```

2. Navigate to the project directory:
    ```bash
    cd /content/Spotting-Suspicious-Citations-/Capstone Project Phase B/Code
    ```

3. Make the shell scripts executable:
    ```bash
    chmod +x ./bash/*.sh
    ```

4. Install Python and pip if they are not already installed:
    ```bash
    sudo apt update
    sudo apt install python3
    sudo apt install python3-pip
    ```

5. Check the versions of Python and pip:
    ```bash
    python3 --version
    pip3 --version
    ```

6. Create a symbolic link for Python:
    ```bash
    sudo ln -s /usr/bin/python3 /usr/bin/python
    ```

7. Install all the dependencies:
    ```bash
    ./bash/install_dependencies.sh
    ```

8. Run the model based on the Cora Palintoid bash script with arguments:
    ```bash
    ./bash/run_cora.sh
    ```

9. Collect all the result files to the results folder:
    ```bash
    ./bash/collectAllResultFiles.sh
    ```

## Project Dependencies

The project requires the following dependencies:

- lightning
- tensorboard
- torch==2.1.2
- torchvision==0.16.2
- torchaudio==2.1.2
- pyg_lib
- torch_scatter
- torch_sparse
- torch_cluster
- torch_spline_conv
- torch_geometric
- cython
- matplotlib
- numpy==1.26.4

## Reuse Policy

The code in this project can be reused for research purposes. If you use this code, please indicate where it was taken from and attach acknowledgments. Proper citation and acknowledgment of the original authors are required.