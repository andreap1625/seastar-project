import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
os.environ.pop("SSL_CERT_FILE", None)
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # ── Configuración ──────────────────────────────────────
    DATA_DIR    = "C:/Users/andre/.cache/kagglehub/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification/versions/1"
    BATCH_SIZE  = 32
    EPOCHS_HEAD = 10   # fase 1: solo la cabeza
    EPOCHS_FINE = 20   # fase 2: fine tuning completo
    LR_HEAD     = 1e-3 # lr más alto para la cabeza nueva
    LR_FINE     = 1e-5 # lr muy bajo para fine tuning
    IMG_SIZE    = 224
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES     = ['bleached_corals', 'healthy_corals']

    print(f"Usando: {DEVICE}")

    # ── Transforms ─────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ── Dataset y split 80/10/10 ───────────────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

    total      = len(full_dataset)
    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    test_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    val_ds = torch.utils.data.Subset(val_dataset, val_ds.indices)
    test_ds = torch.utils.data.Subset(test_dataset, test_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # ── Modelo: quitar cabeza original y congelar backbone ─
    backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    backbone.classifier = nn.Identity()
    # num_classes=0 elimina la capa clasificadora original

    # Congelar todo el backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Obtener tamaño de salida del backbone
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        features_dim = backbone(dummy).shape[1]
    print(f"Dimensión de features del backbone: {features_dim}")

    # Cabeza de clasificación personalizada
    classification_head = nn.Sequential(
        nn.Linear(features_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 2)
    )

    # Modelo completo
    class CoralClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head     = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = CoralClassifier(backbone, classification_head).to(DEVICE)

    # ── Función de entrenamiento por epoch ─────────────────
    def run_epoch(loader, training=True):
        if training:
            model.train()
        else:
            model.eval()

        total_loss, correct, total = 0, 0, 0
        with torch.set_grad_enabled(training):
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if training:
                    optimizer.zero_grad()
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += labels.size(0)

        return total_loss / len(loader), correct / total

    criterion = nn.CrossEntropyLoss()

    # ══════════════════════════════════════════════════════
    # FASE 1 — Solo entrenar la cabeza (backbone congelado)
    # ══════════════════════════════════════════════════════
    print("\n── FASE 1: Entrenando cabeza de clasificación ──")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    all_train_losses, all_val_losses, all_val_accs = [], [], []

    for epoch in range(EPOCHS_HEAD):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss,   val_acc   = run_epoch(val_loader,   training=False)

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnet.pth")

        print(f"Epoch {epoch+1:02d}/{EPOCHS_HEAD} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ══════════════════════════════════════════════════════
    # FASE 2 — Descongelar backbone y fine tuning completo
    # ══════════════════════════════════════════════════════
    print("\n── FASE 2: Fine tuning completo ──")

    # Descongelar backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # LR muy bajo para no destruir los pesos preentrenados
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(EPOCHS_FINE):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss,   val_acc   = run_epoch(val_loader,   training=False)

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnet.pth")

        print(f"Epoch {epoch+1:02d}/{EPOCHS_FINE} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ── Evaluación final en test ───────────────────────────
    model.load_state_dict(torch.load("best_efficientnet.pth"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs  = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("\n--- MÉTRICAS FINALES ---")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # ── Curvas de entrenamiento ────────────────────────────
    epochs_range = range(1, EPOCHS_HEAD + EPOCHS_FINE + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, all_train_losses, label='Train Loss')
    ax1.plot(epochs_range, all_val_losses,   label='Val Loss')
    ax1.axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Inicio Fine Tuning')
    ax1.set_title('Loss por Epoch')
    ax1.legend()

    ax2.plot(epochs_range, all_val_accs, label='Val Accuracy', color='green')
    ax2.axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Inicio Fine Tuning')
    ax2.set_title('Accuracy de Validación')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    # ── Matriz de confusión ────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Matriz de Confusión - EfficientNet-B0')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print(f"\nMejor Val Accuracy: {best_val_acc:.4f}")
    print("Archivos guardados: best_efficientnet.pth | confusion_matrix.png | training_curves.png")