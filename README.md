# Multimodal Self-Supervised Image Super-resolution and Registration for Remote Sensing Imagery

## Overview

SR-FloW-Net is a deep learning model designed for the registration of multispectral images. This model aligns images taken from different spectral bands, enabling accurate and efficient analysis of multispectral data. The SR-FloW-Net architecture leverages the power of convolutional neural networks (CNNs) to learn complex spatial transformations and align images with high precision.

## Features

- **End-to-End Training**: Train the model from scratch with your multispectral datasets.
- **High Accuracy**: Achieve precise alignment of images across different spectral bands.
- **Flexible Architecture**: Easily adaptable to various multispectral imaging applications.
- **Comprehensive Documentation**: Detailed instructions and examples for easy implementation and customization.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/SergiFuster/SR-FloW-Net.git
cd SR-FloW-Net
```

Create virtual environment:
```
python -m venv myvenv
```

Initialize virtual environment:
```
./myvenv/Scripts/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

To avoid installation problems torch is going to be installed independently with:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

All in one:
```
git clone https://github.com/SergiFuster/SR-FloW-Net.git
cd SR-FloW-Net
python -m venv myvenv
./myvenv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

## Train a Model

A script has already been prepared to run after setting up the virtual environment. Ensure that the images from the dataset mentioned below are placed in the working directory.

The script is designed to take images from the following structure:

```
data
└── images
│   ├── S2
│   │   └── 18.mat
│   │
│   └── S3
│       └── 18.mat
```

Once the working directory is set up, you can execute the script by running the following command in the command line (cmd) or by using an IDE such as [Visual Studio Code](https://code.visualstudio.com/):

```bash
python test.py
```

**Note:** The predefined number of epochs is set to 1000, which is the minimum required to achieve satisfactory results.

## Check Saved Models

If you have trained models and saved them in your working directory, you might be interested in using the embedded Python application included in this repository.

To launch the application, run the `app.py` script by executing the following command:

```bash
python app.py
```

Next, open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

All files with a `.pth` extension will be displayed. By clicking on a valid file, you will see the model information for each training session in JSON format, as illustrated in the example below:

```json
{
  "training": [
    {
      "epochs": 1000,
      "image": "model_name",
      "learning_rate": 0.001,
      "loss": -0.491024579178,
      "loss_function": "CC3D",
      "time": 367.730907678604126,
      "weights": [
        1,
        0.5,
        0.5
      ]
    }
  ]
}
```

## Dataset
The dataset used for our experiments can be found at [DATASET](https://ujies.sharepoint.com/:f:/s/Z365-EVP/El3yqTw3pypNsnYzgT5RFqYBxXAUREIgePaopyoqODYMmg?e=c7biWg)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact [sfuster@uji.es](mailto:sfuster@uji.es).
