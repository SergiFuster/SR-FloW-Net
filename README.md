# SR-FloW-Net: Multimodal Self-Supervised Image Super-resolution and Registration for Remote Sensing Imagery

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
## Dataset
The dataset used for our experiments can be found at [DATASET](https://ujies.sharepoint.com/sites/Z365-EVP/Documents%20compartits/Forms/AllItems.aspx?id=%2Fsites%2FZ365%2DEVP%2FDocuments%20compartits%2FPublic%2DData%2FS2%20S3%20DATASET%20FLOU%2DNET&p=true&ga=1)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact [sfuster@uji.es](mailto:sfuster@uji.es).
