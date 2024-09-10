# SR-FloW-Net: AI Model for Multispectral Image Registration and Super-Resolution

## Overview

FloU-Net is a deep learning model designed for the registration of multispectral images. This model aligns images taken from different spectral bands, enabling accurate and efficient analysis of multispectral data. The FloU-Net architecture leverages the power of convolutional neural networks (CNNs) to learn complex spatial transformations and align images with high precision.

## Features

- **End-to-End Training**: Train the model from scratch with your multispectral datasets.
- **High Accuracy**: Achieve precise alignment of images across different spectral bands.
- **Flexible Architecture**: Easily adaptable to various multispectral imaging applications.
- **Comprehensive Documentation**: Detailed instructions and examples for easy implementation and customization.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/SergiFuster/FloU-Net.git
cd FloU-Net
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Prepare your multispectral image dataset and organize it into the following structure:

```
data/
├── images/
│   ├── mat/
│   │   ├── S2
│   │   ├── S3
```

Run the training script:

```bash
python train.py --data_dir data --epochs 50 --batch_size 16
```

The complete dataset and information of every coupled Sentinel-2/Sentinel-3 pair used in the experimentation can be found in: [DATASET](https://ujies.sharepoint.com/sites/Z365-EVIS/Documents%20compartits/Forms/AllItems.aspx?csf=1&web=1&e=BCECgp&CID=39444ccc%2Df63d%2D4b61%2D82e2%2Da7f9ef5f60b7&FolderCTID=0x012000830B65DFB949E748998958F7F3AA09D9&id=%2Fsites%2FZ365%2DEVIS%2FDocuments%20compartits%2FPublic%5FData%2FS2%20S3%20DATASET%20FLOU%2DNET)

### Inference

Use the trained model to perform image registration on new data:

```bash
python inference.py --model_path path/to/trained_model.pth --input_dir path/to/input_images --output_dir path/to/output_images
```

## Evaluation

Evaluate the model performance on the validation set:

```bash
python evaluate.py --data_dir dataset/val --model_path path/to/trained_model.pth
```

## Results

Visualize the registration results with the provided visualization tools:

```bash
python visualize.py --input_dir path/to/input_images --output_dir path/to/output_images
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact [sfuster@uji.es](mailto:sfuster@uji.es).
