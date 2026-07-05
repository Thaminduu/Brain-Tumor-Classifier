import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
	model = keras.Sequential(
		[
			keras.layers.Input(shape=input_shape),
			keras.layers.RandomFlip("horizontal"),
			keras.layers.RandomRotation(0.08),
			keras.layers.RandomZoom(0.1),
			keras.layers.Rescaling(1.0 / 255.0),
			keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
			keras.layers.BatchNormalization(),
			keras.layers.Activation("relu"),
			keras.layers.MaxPooling2D(),
			keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
			keras.layers.BatchNormalization(),
			keras.layers.Activation("relu"),
			keras.layers.MaxPooling2D(),
			keras.layers.Conv2D(128, 3, padding="same", use_bias=False),
			keras.layers.BatchNormalization(),
			keras.layers.Activation("relu"),
			keras.layers.MaxPooling2D(),
			keras.layers.Flatten(),
			keras.layers.Dense(256, use_bias=False),
			keras.layers.BatchNormalization(),
			keras.layers.Activation("relu"),
			keras.layers.Dropout(0.4),
			keras.layers.Dense(num_classes, activation="softmax"),
		],
		name="brain_tumor_classifier",
	)
	return model


def load_image_dataset(
	data_dir: Path,
	image_size: tuple[int, int],
	batch_size: int,
	seed: int,
) -> tuple[object, list[str]]:
	dataset = keras.utils.image_dataset_from_directory(
		data_dir,
		labels="inferred",
		label_mode="categorical",
		image_size=image_size,
		batch_size=batch_size,
		seed=seed,
	)

	class_names = dataset.class_names
	autotune = tf.data.AUTOTUNE
	dataset = dataset.prefetch(autotune)
	return dataset, class_names


def save_history_plot(history: keras.callbacks.History, output_path: Path, show_plot: bool) -> None:
	acc = history.history.get("accuracy", [])
	val_acc = history.history.get("val_accuracy", [])
	loss = history.history.get("loss", [])
	val_loss = history.history.get("val_loss", [])

	if not loss:
		return

	epochs = range(1, len(loss) + 1)
	plt.figure(figsize=(12, 4))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, acc, marker="o", label="Train")
	if val_acc:
		plt.plot(epochs, val_acc, marker="o", label="Validation")
	plt.title("Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, loss, marker="o", label="Train")
	if val_loss:
		plt.plot(epochs, val_loss, marker="o", label="Validation")
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.grid(True, alpha=0.3)
	plt.legend()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	print(f"Saved training plot to: {output_path}")
	if show_plot:
		plt.show()
	plt.close()


def train(args: argparse.Namespace) -> None:
	keras.utils.set_random_seed(args.seed)

	train_dir = Path(args.data_dir) / "training_set"
	test_dir = Path(args.data_dir) / "testing_set"

	if not train_dir.exists():
		raise FileNotFoundError(f"Training directory not found: {train_dir}")
	if not test_dir.exists():
		raise FileNotFoundError(f"Testing directory not found: {test_dir}")

	train_ds, class_names = load_image_dataset(
		data_dir=train_dir,
		image_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		seed=args.seed,
	)
	test_ds, _ = load_image_dataset(
		data_dir=test_dir,
		image_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		seed=args.seed,
	)

	num_classes = len(class_names)
	model = build_model(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)

	model.summary()

	callbacks: list[keras.callbacks.Callback] = [
		keras.callbacks.EarlyStopping(
			monitor="val_accuracy",
			patience=5,
			restore_best_weights=True,
		)
	]

	history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=callbacks)

	test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
	print(f"\nTest loss: {test_loss:.4f}")
	print(f"Test accuracy: {test_accuracy:.4f}")

	model_path = Path(args.output_model)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	model.save(model_path)
	print(f"Saved model to: {model_path}")

	labels_path = model_path.with_suffix(".labels.txt")
	labels_path.write_text("\n".join(class_names), encoding="utf-8")
	print(f"Saved class labels to: {labels_path}")

	save_history_plot(history, Path(args.plot_path), args.show_plot)


def load_labels(model_path: Path) -> list[str]:
	labels_path = model_path.with_suffix(".labels.txt")
	if labels_path.exists():
		return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
	return ["class_0", "class_1", "class_2", "class_3"]


def preprocess_for_prediction(image_path: Path, img_size: int) -> np.ndarray:
	image = Image.open(image_path).convert("RGB")
	image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
	image_array = np.asarray(image, dtype="float32")
	return np.expand_dims(image_array, axis=0)


def predict(args: argparse.Namespace) -> None:
	model_path = Path(args.model_path)
	image_path = Path(args.predict_image)

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	model = keras.models.load_model(model_path)
	labels = load_labels(model_path)
	if len(labels) != 4:
		raise ValueError("Expected four class labels for multi-class model prediction.")

	batch = preprocess_for_prediction(image_path, args.img_size)
	probabilities = model.predict(batch, verbose=0)[0]

	predicted_class_idx = int(np.argmax(probabilities))
	predicted_label = labels[predicted_class_idx]
	confidence = float(probabilities[predicted_class_idx])

	print(f"\nPredicted tumor type: {predicted_label}")
	print(f"Confidence: {confidence:.2%}\n")
	print("Class probabilities:")
	for i, (label, prob) in enumerate(zip(labels, probabilities)):
		print(f"  {label}: {prob:.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train or run a brain tumor image classifier.")
	parser.add_argument("--data-dir", type=str, default="data", help="Root folder with training_set and testing_set subfolders.")
	parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
	parser.add_argument("--img-size", type=int, default=224, help="Square image size for training and prediction.")
	parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--output-model", type=str, default="artifacts/brain_tumor_model.keras", help="Saved model path.")
	parser.add_argument("--plot-path", type=str, default="artifacts/brain_tumor_training_plot.png", help="Path for training curves image.")
	parser.add_argument("--show-plot", action="store_true", help="Show training plot window.")

	parser.add_argument("--predict-image", type=str, default=None, help="Path to a single image for prediction mode.")
	parser.add_argument("--model-path", type=str, default="artifacts/brain_tumor_model.keras", help="Model path for prediction mode.")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.predict_image:
		predict(args)
	else:
		train(args)
