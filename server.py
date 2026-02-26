# server.py 
import os
import io
import json
import time
import pickle
import joblib
import asyncio
import cv2
import traceback
from PIL import Image
from typing import Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# ------------- Configuration & utils -----------------
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

# Simple websocket broadcast for logs
clients: set[WebSocket] = set()
log_queue: asyncio.Queue = asyncio.Queue()


async def broadcast_log(obj: dict):
    """Put message on a queue for the broadcaster to send."""
    try:
        text = json.dumps(obj)
        await log_queue.put(text)
    except Exception:
        # ensure we never crash logging
        print("broadcast_log error", traceback.format_exc())


async def _log_broadcaster():
    """Background task that forwards queued log messages to all connected clients."""
    while True:
        msg = await log_queue.get()
        # iterate over a snapshot
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                try:
                    await ws.close()
                except:
                    pass
                clients.discard(ws)


@app.on_event("startup")
async def startup_event():
    # Launch the broadcaster task on startup (safer under uvicorn)
    asyncio.create_task(_log_broadcaster())


def save_npz(X, y):
    path = os.path.join(ARTIFACT_DIR, "dataset.npz")
    np.savez_compressed(path, X=X, y=y)
    return path


def load_npz():
    path = os.path.join(ARTIFACT_DIR, "dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found. Call /load_dataset first.")
    d = np.load(path)
    return d["X"], d["y"]


def save_pickle(obj, name):
    path = os.path.join(ARTIFACT_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(name):
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found.")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------- FastAPI models ----------------------
class SplitParams(BaseModel):
    test_size: float


class TrainParams(BaseModel):
    model_name: str  # 'svm_pca','svm','knn','knn_pca','cnn'
    epochs: Optional[int] = 5  # only for cnn
    knn_k: Optional[int] = 5


# -------------- Endpoints --------------------------

@app.post("/load_dataset")
async def load_dataset():
    """
    Load MNIST (handwritten) dataset into X,y and save artifacts/dataset.npz.
    X will be flattened images (n_samples, 28*28) as float32 normalized to [0,1].
    """
    try:
        await broadcast_log({"type": "info", "message": "Loading MNIST dataset..."})
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        # combine train+test to allow arbitrary split later
        X_train = mnist_train.data.numpy()  # uint8 HxW
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy()
        y_test = mnist_test.targets.numpy()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        # keep original shape (N,28,28) saved but we will flatten in preprocess
        # Flatten and scale to [0,1] now or later? We keep raw pixels normalized.
        X = X.reshape((X.shape[0], -1)).astype(np.float32) / 255.0
        save_npz(X, y)
        await broadcast_log({"type": "step", "step": "data_loaded", "message": f"Loaded MNIST: {X.shape[0]} samples."})
        return {"status": "ok", "n_samples": int(X.shape[0])}
    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Error loading dataset: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ----------------------------------------------------------
# PREPROCESS (create both raw scaled and PCA datasets)
# ----------------------------------------------------------
@app.post("/preprocess")
async def preprocess_dataset():
    """
    ALWAYS create:
      - train_test_raw.npz   (scaled full 784 features)  -> used by svm/knn
      - train_test_pca.npz   (PCA 50-d features)         -> used by svm_pca/knn_pca
      - scaler.pkl (StandardScaler fitted on full X)
      - pca.pkl (PCA fitted on training split)
    """
    try:
        await broadcast_log({"type": "info", "message": "Starting preprocessing..."})
        # Load dataset saved by /load_dataset
        X, y = load_npz()  # will raise if missing

        # X is already flattened and normalized to [0,1] in load_dataset
        # Apply standard scaling to features (important for SVM/KNN/PCA)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        save_pickle(scaler, "scaler.pkl")
        await broadcast_log({"type": "info", "message": "StandardScaler fitted and saved."})

        # Split into train/test with stratify
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save RAW scaled train/test (784 dims)
        raw_path = os.path.join(ARTIFACT_DIR, "train_test_raw.npz")
        np.savez_compressed(raw_path,
                            X_train=X_train_raw, X_test=X_test_raw,
                            y_train=y_train, y_test=y_test)
        await broadcast_log({"type": "step", "step": "preprocess", "message": f"Saved raw train/test to {raw_path}", "raw_shape": list(X_train_raw.shape)})

        # Fit PCA on training raw (scaled) features and transform both train and test
        pca = PCA(n_components=50, random_state=42)
        pca.fit(X_train_raw)
        X_train_pca = pca.transform(X_train_raw)
        X_test_pca = pca.transform(X_test_raw)

        # Save PCA dataset
        pca_path = os.path.join(ARTIFACT_DIR, "train_test_pca.npz")
        np.savez_compressed(pca_path,
                            X_train=X_train_pca, X_test=X_test_pca,
                            y_train=y_train, y_test=y_test)
        save_pickle(pca, "pca.pkl")
        await broadcast_log({"type": "step", "step": "preprocess", "message": f"Saved PCA train/test to {pca_path}", "pca_shape": list(X_train_pca.shape)})

        return {"status": "ok", "raw_shape": X_train_raw.shape, "pca_shape": X_train_pca.shape}
    except FileNotFoundError as fe:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Preprocess error: {str(fe)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(fe)}, status_code=400)
    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Error in preprocessing: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.post("/split")
async def split(params: SplitParams):
    """
    Backwards-compatible split endpoint that will split whichever preprocessed X exists.
    If train_test_raw.npz exists, split and save a generic train_test.npz for legacy endpoints.
    """
    try:
        test_size = float(params.test_size)

        raw_path = os.path.join(ARTIFACT_DIR, "train_test_raw.npz")
        pca_path = os.path.join(ARTIFACT_DIR, "train_test_pca.npz")

        if os.path.exists(raw_path):
            d = np.load(raw_path)
            X = np.concatenate([d["X_train"], d["X_test"]], axis=0)
            y = np.concatenate([d["y_train"], d["y_test"]], axis=0)
            await broadcast_log({"type": "info", "message": "Using train_test_raw.npz for splitting."})
        else:
            # fallback to dataset.npz
            X, y = load_npz()
            await broadcast_log({"type": "info", "message": "Using raw dataset.npz for split (no preprocessed found)."})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        np.savez_compressed(os.path.join(ARTIFACT_DIR, "train_test.npz"), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        await broadcast_log({"type": "step", "step": "split", "message": f"Train/test split done. test_size={test_size}", "train_shape": list(X_train.shape), "test_shape": list(X_test.shape)})
        return {"status": "ok", "train_shape": X_train.shape, "test_shape": X_test.shape}
    except FileNotFoundError as fe:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Split error: {str(fe)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(fe)}, status_code=400)
    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Error splitting: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ----------------- Helper for plotting -----------------
def save_confusion_matrix(y_true, y_pred, outname="confusion.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(cax, ax=ax)
    ticks = np.arange(len(np.unique(y_true)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    out_path = os.path.join(ARTIFACT_DIR, outname)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_classification_report_image(report_text, outname="class_report.txt"):
    """
    Save the classification report in text and also render it to a PNG image (monospace).
    This makes it easy to display in Flutter as an image.
    """
    txt_path = os.path.join(ARTIFACT_DIR, outname)
    with open(txt_path, "w") as f:
        f.write(report_text)

    # Prepare a readable text image
    lines = report_text.splitlines()
    # configure figure size to fit text (approx)
    height = max(6, 0.25 * len(lines))
    fig, ax = plt.subplots(figsize=(8, height))
    ax.axis("off")
    # Draw each line with monospace font
    y = 1.0
    line_y_step = 1.0 / (len(lines) + 1)
    for i, line in enumerate(lines):
        ax.text(0.01, 1 - i * 0.02, line, fontsize=9, fontfamily="monospace", va="top")
    img_path = os.path.join(ARTIFACT_DIR, outname.replace(".txt", ".png"))
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return txt_path, img_path


# ----------------- Model training functions -----------------
def train_sklearn_model(model_name: str, knn_k: int = 5):

    # Choose dataset depending on model type
    if model_name in ("svm_pca", "knn_pca"):
        file_path = os.path.join(ARTIFACT_DIR, "train_test_pca.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Run /preprocess first.")
        d = np.load(file_path)
    else:
        file_path = os.path.join(ARTIFACT_DIR, "train_test_raw.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Run /preprocess first.")
        d = np.load(file_path)

    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]

    # -------------------------------------------------
    # Train SVM / KNN
    # -------------------------------------------------
    if model_name in ("svm", "svm_pca"):
        clf = svm.SVC(kernel="rbf", probability=False, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join(ARTIFACT_DIR, f"{model_name}_model.joblib"))
        model_path = os.path.join(ARTIFACT_DIR, f"{model_name}_model.joblib")

    elif model_name in ("knn", "knn_pca"):
        clf = neighbors.KNeighborsClassifier(n_neighbors=knn_k)
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join(ARTIFACT_DIR, f"{model_name}_model.joblib"))
        model_path = os.path.join(ARTIFACT_DIR, f"{model_name}_model.joblib")

    else:
        raise ValueError("Unknown sklearn model: " + model_name)

    # ---------------- Evaluation ----------------
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    report = classification_report(y_test, preds)

    # Save predictions & metrics
    np.savez_compressed(os.path.join(ARTIFACT_DIR, f"{model_name}_preds.npz"),
                        y_test=y_test, y_pred=preds)

    with open(os.path.join(ARTIFACT_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "report": report
        }, f)

    # Save confusion matrix & report image
    cm_path = save_confusion_matrix(y_test, preds, outname=f"{model_name}_confusion.png")
    _, report_img = save_classification_report_image(
        report, outname=f"{model_name}_classification_report.txt"
    )

    return {
        "model_path": model_path,
        "confusion_path": cm_path,
        "report_img": report_img
    }


# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_cnn_sync(epochs=5, batch_size=64, lr=1e-3):
    """
    Synchronous CNN training used by async wrapper.
    Saves model state, metrics, predictions, and plots to artifacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(1, epochs + 1):
        # notify via broadcast (from running async wrapper)
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        for i, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / n_batches
        train_losses.append(avg_loss)

    # evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(target.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_test = np.concatenate(all_targets, axis=0)
    acc = float(accuracy_score(y_test, y_pred))
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred)

    # save model & metrics
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "cnn_model.pth"))
    np.savez_compressed(os.path.join(ARTIFACT_DIR, "cnn_preds.npz"), y_test=y_test, y_pred=y_pred)
    with open(os.path.join(ARTIFACT_DIR, "cnn_metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": report}, f)
    # loss plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("CNN Training Loss")
    loss_path = os.path.join(ARTIFACT_DIR, "cnn_loss.png")
    fig.savefig(loss_path)
    plt.close(fig)
    cm_path = save_confusion_matrix(y_test, y_pred, outname="cnn_confusion.png")
    report_txt, report_img = save_classification_report_image(report, outname="cnn_classification_report.txt")
    return {"model_path": os.path.join(ARTIFACT_DIR, "cnn_model.pth"), "confusion_path": cm_path, "report_img": report_img, "loss_plot": loss_path}


@app.post("/train")
async def train(params: TrainParams, background_tasks: BackgroundTasks):
    """
    Train a chosen model.
    model_name: 'svm_pca','svm','knn','knn_pca','cnn'
    This endpoint will start training and save artifacts to disk. It will NOT return predictions or metrics directly in the response.
    Use /results?model_name=<name> to fetch metrics & image paths after training completes.
    """
    try:
        model_name = params.model_name
        await broadcast_log({"type": "info", "message": f"Training requested: {model_name}"})
        if model_name == "cnn":
            # Run CNN training in a background thread but emit logs via broadcast_log from this coroutine.
            import threading

            def cnn_worker(epochs):
                try:
                    asyncio.run(_async_train_cnn_wrapper(epochs))
                except Exception as e:
                    asyncio.run(broadcast_log({"type": "error", "message": f"CNN worker error: {repr(e)}"}))

            t = threading.Thread(target=cnn_worker, args=(params.epochs or 5,), daemon=True)
            t.start()
            return {"status": "started", "model": model_name}
        else:
            # sklearn models: train synchronously (blocking) but do NOT return metrics in response
            await broadcast_log({"type": "info", "message": f"Training {model_name} (sklearn) started"})
            res = train_sklearn_model(model_name, knn_k=params.knn_k or 5)
            await broadcast_log({"type": "done", "message": f"{model_name} training finished and artifacts saved."})
            # Return minimal info only
            return {"status": "ok", "model": model_name, "artifacts_saved": True}
    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Training error: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


async def _async_train_cnn_wrapper(epochs):
    """
    Async wrapper that emits progress logs while running the synchronous training function.
    """
    await broadcast_log({"type": "info", "message": f"Starting CNN training (epochs={epochs})"})
    # We will periodically post progress messages while calling the sync trainer in a thread
    loop = asyncio.get_event_loop()

    def run_sync():
        return train_cnn_sync(epochs=epochs)

    # Run the synchronous train in a thread to avoid blocking the event loop
    res = await loop.run_in_executor(None, run_sync)
    await broadcast_log({"type": "done", "message": f"CNN training finished. Artifacts saved."})
    return res


@app.post("/predict")
async def predict(model_name: str = Form(...)):
    """
    For sklearn models and cnn, load saved predictions (if available) and return simple aggregated results.
    This endpoint is intended for explicit 'Predict' action from the UI, not automatically called during training.
    """
    try:
        if model_name == "cnn":
            preds_path = os.path.join(ARTIFACT_DIR, "cnn_preds.npz")
            if not os.path.exists(preds_path):
                raise FileNotFoundError("CNN predictions not found. Train CNN first.")
            d = np.load(preds_path)
            y_test, y_pred = d["y_test"], d["y_pred"]
        else:
            preds_path = os.path.join(ARTIFACT_DIR, f"{model_name}_preds.npz")
            if not os.path.exists(preds_path):
                raise FileNotFoundError(f"{preds_path} not found. Train {model_name} first.")
            d = np.load(preds_path)
            y_test, y_pred = d["y_test"], d["y_pred"]

        acc = float(accuracy_score(y_test, y_pred))
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        await broadcast_log({"type": "done", "message": f"Predict completed for {model_name}. Acc={acc:.4f}"})

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        return {"status": "ok", "metrics": metrics}

    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Predict error: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/results")
async def results(model_name: str = Query(...)):
    """
    Return metrics JSON and artifact filenames for the requested model.
    Frontend should call /results?model_name=svm_pca (or svm, knn, knn_pca, cnn)
    This endpoint does not compute metrics; it returns saved artifacts.
    """
    try:
        metrics_file = os.path.join(ARTIFACT_DIR, f"{model_name}_metrics.json")
        confusion_file = os.path.join(ARTIFACT_DIR, f"{model_name}_confusion.png")
        report_img = os.path.join(ARTIFACT_DIR, f"{model_name}_classification_report.png")
        loss_plot = os.path.join(ARTIFACT_DIR, f"{model_name}_loss.png")  # mostly for cnn

        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        artifacts = {}
        # Use web-accessible paths under /artifacts/
        if os.path.exists(confusion_file):
            artifacts["confusion_img"] = f"/artifacts/{os.path.basename(confusion_file)}"
        if os.path.exists(report_img):
            artifacts["report_img"] = f"/artifacts/{os.path.basename(report_img)}"
        # cnn loss plot
        if os.path.exists(loss_plot):
            artifacts["loss_plot"] = f"/artifacts/{os.path.basename(loss_plot)}"

        return {"status": "ok", "model": model_name, "metrics": metrics, "artifacts": artifacts}
    except Exception as e:
        tb = traceback.format_exc()
        await broadcast_log({"type": "error", "message": f"Results error: {str(e)}", "trace": tb})
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """Serve an artifact file such as images or metrics. (Fallback, StaticFiles mounted handles most)"""
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "detail": "Not found"}, status_code=404)
    # For text metrics, return JSON if possible
    if filename.endswith(".json"):
        with open(path, "r") as f:
            return JSONResponse(json.load(f))
    return FileResponse(path)


@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket):
    """
    WebSocket endpoint for streaming logs.
    Connect from Flutter and receive JSON messages:
    {"type":"info"/"progress"/"epoch"/"done"/"error", "message": "..."}
    """
    await ws.accept()
    clients.add(ws)
    client_id = id(ws)
    await broadcast_log({"type": "info", "message": f"Client {client_id} connected to logs."})
    try:
        while True:
            # keep connection alive; client may send pings (ignore contents)
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore unexpected messages; continue
                await asyncio.sleep(0.1)
    finally:
        clients.discard(ws)
        await broadcast_log({"type": "info", "message": f"Client {client_id} disconnected."})


        @app.get("/")
        async def root():
            return {"message": "MNIST training pipeline backend. Use REST endpoints and WebSocket /ws/logs for live updates."}

        @app.post("/predict_single")
        async def predict_single(model_name: str = Form(...), file: UploadFile = File(...)):
            """
            Predict digit from uploaded image.
            Returns: prediction + saved image path
            """
            try:
                contents = await file.read()

                npimg = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    return JSONResponse(
                        {"status": "error", "detail": "Invalid image"}, status_code=400
                    )

                img_resized = cv2.resize(img, (28, 28))
                img_flat = img_resized.reshape(1, 784) / 255.0

                model_path = os.path.join(ARTIFACT_DIR, f"{model_name}_model.joblib")
                if not os.path.exists(model_path):
                    return JSONResponse(
                        {"status": "error", "detail": f"Model {model_name} not trained"},
                        status_code=400,
                    )

                model = joblib.load(model_path)
                pred = int(model.predict(img_flat)[0])

                save_path = os.path.join(ARTIFACT_DIR, "uploaded_digit.png")
                cv2.imwrite(save_path, img_resized)

                return {
                    "status": "ok",
                    "prediction": pred,
                    "image_url": "/artifacts/uploaded_digit.png"
                }

            except Exception as e:
                return JSONResponse(
                    {"status": "error", "detail": str(e)}, status_code=500
                )
# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


