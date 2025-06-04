# SantaVision - CNN Model for Recognizing Christmas Eve Dishes
## Project Overview
This project was developed during the *Noc Sztucznej Inteligencji* hackathon to create a artificial intelligence model for classifying images of traditional Polish Christmas Eve dishes. The model identifies 8 dish categories: barszcz czerwony, bigos, kutia, makowiec, pierniki, pierogi, sernik, and zupa grzybowa, achieving a test F1-score of 0.9835 and a test loss of 0.0921.
## Authors
- Jakub Zdancewicz
- Wiktor Niedźwiedzki

## Features
- **Model Architecture**: Fine-tuned ResNet-50 with a modified fully connected layer to classify 8 dish categories.
- **Dataset**: Custom dataset created and cleaned using duckduckgo-search of Christmas Eve dish images, split into train (70%), validation (20%), and test (10%) sets.
- **Training**: Utilized Adam optimizer with learning rate decay and cross-entropy loss, trained for 11 epochs on Google Colab with GPU support.
- **Data Preprocessing**: Images resized to 224x224, converted from RGBA to RGB where necessary, and loaded using PyTorch’s ImageFolder and DataLoader.
- **Evaluation**: Achieved high performance with a test F1-score of 0.9835 and test loss of 0.0921, evaluated using weighted F1-score and loss metrics.
- **Inference**: Implemented a pipeline for classifying new images, mapping predictions to human-readable dish names.

## Requirements
- Python 3.10
- PyTorch
- torchvision
- scikit-learn
- PIL (Pillow)
- Google Colab (for training with GPU support)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pillow
   ```
3. Download and unzip the dataset (`data.zip`) into the project directory.

## Usage
1. **Prepare the Dataset**:
   - Run the data preparation script to split images into train, validation, and test sets:
     ```python
     database_path = "data"
     splitted_data_path = "m"
     image_datasets, dataloaders = prepare_data(database_path, splitted_data_path, validation_split=0.2, batch_size=128)
     ```
2. **Train the Model**:
   - Execute the training script with the fine-tuned ResNet-50 model:
     ```python
     model_ft = models.resnet50(weights='DEFAULT')
     num_ftrs = model_ft.fc.in_features
     model_ft.fc = nn.Linear(num_ftrs, num_classes)
     optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
     model_ft = train_model(model_ft, dataloaders, image_datasets, device, loss_fn, optimizer_ft, num_epochs=11)
     ```
3. **Test the Model**:
   - Evaluate on the test set:
     ```python
     model_ft.eval()
     # Run test loop to compute loss and F1-score
     ```
4. **Classify New Images**:
   - Use the inference function to classify new images:
     ```python
     predicted_label = classify_image(model_ft, "path/to/image.jpg", class_labels, device)
     print(f"Predicted dish: {class_labels_reverse[predicted_label]}")
     ```

## Results
- **Training**: Achieved a train F1-score of 0.9982 and loss of 0.0192 after 11 epochs.
- **Validation**: Reached a validation F1-score of 0.9750 and loss of 0.0806.
- **Test**: Obtained a test F1-score of 0.9835 and loss of 0.0921, demonstrating robust generalization.

## Future Improvements
- Apply data augmentation (e.g., random rotations, flips) to improve model robustness.
- Experiment with other architectures.
- Expand the dataset with more diverse images to enhance model accuracy.
