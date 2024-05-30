import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataloader.dataloader import get_train_loader, get_val_loader
from models.rgb_segmentation import RGBNet
# from models.noise_segmentation import NoiseNet
from timm.scheduler.step_lr import StepLRScheduler
import argparse
from utils import RunningAggregation
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default=1,
        choices=[1, 2],
        type=int,
        help="Sets the training phase: 1 for segmentation, 2 for classification",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs to train the model for",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = RGBNet().to(DEVICE)
    # model = NoiseNet().to(DEVICE)
    checkpoint = torch.load("runs/Apr02_03-01-56_20f480a63185/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    #model.init_weights("pretrained/mit_b2.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
    scheduler = StepLRScheduler(optimizer, decay_t = 20, decay_rate=0.1)

    if args.phase == 1:
        from models.losses import Phase1Loss
        loss_fn = Phase1Loss()
        
        # freeze confidence decoder and classification head
        for p in model.confidence_decoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False
            
    elif args.phase == 2:
        from models.losses import Phase2Loss
        loss_fn = Phase2Loss()
        
        # freeze encoder and localisation decoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False

    print(f"Training started for {args.epochs} epochs.")
    writer = SummaryWriter()
    best_metric = -1.0
    for i in range(args.epochs):
        print(f"Epoch: {i + 1}/{args.epochs}")

        # Training
        model.train()
        train_loss, train_metric = RunningAggregation(), RunningAggregation()
        pbar = tqdm(train_loader)
        pbar.set_description("Training...")
        for images, targets in pbar:
            images = images.to(DEVICE)
            masks = targets["masks"].to(DEVICE)
            labels = targets["labels"].to(DEVICE)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            logits_mask, conf_map, logits_class = model(images)
            batch_loss = loss_fn(logits_mask, conf_map, logits_class, masks, labels)
            train_loss.add(batch_loss.item() * batch_size, n=batch_size)
            batch_loss.backward()
            optimizer.step()

            if args.phase == 1:
                stats = smp.metrics.get_stats(
                    output=logits_mask,
                    target=(masks > 0.5).int(),
                    mode="binary",
                    threshold=0.5,
                )
                pixel_f1 = smp.metrics.f1_score(*stats, reduction="micro").item()
                train_metric.add(pixel_f1 * batch_size, n=batch_size)
            elif args.phase == 2:
                pred_class = (logits_class > 0).float()
                accuracy = torch.mean((pred_class == labels).float()).item()
                train_metric.add(accuracy * batch_size, n=batch_size)

        if scheduler:
            scheduler.step(i)
        writer.add_scalar("loss", train_loss(), i + 1)
        writer.add_scalar("f1_score" if args.phase == 1 else "accuracy", train_metric(), i + 1)

        # Validation
        model.eval()
        val_loss, val_metric = RunningAggregation(), RunningAggregation()
        pbar = tqdm(val_loader)
        pbar.set_description("Validating...")
        for images, targets in pbar:
            images = images.to(DEVICE)
            masks = targets["masks"].to(DEVICE)
            labels = targets["labels"].to(DEVICE)
            batch_size = images.size(0)

            with torch.no_grad():
                logits_mask, conf_map, logits_class = model(images)
                batch_loss = loss_fn(logits_mask, conf_map, logits_class, masks, labels)
                val_loss.add(batch_loss.item() * batch_size, n=batch_size)
                
                if args.phase == 1:
                    stats = smp.metrics.get_stats(
                        output=logits_mask,
                        target=(masks > 0.5).int(),
                        mode="binary",
                        threshold=0.5,
                    )
                    pixel_f1 = smp.metrics.f1_score(*stats, reduction="micro").item()
                    val_metric.add(pixel_f1 * batch_size, n=batch_size)
                elif args.phase == 2:
                    pred_class = (logits_class > 0).float()
                    accuracy = torch.mean((pred_class == labels).float()).item()
                    val_metric.add(accuracy * batch_size, n=batch_size)

        writer.add_scalar("val_loss", val_loss(), i + 1)
        writer.add_scalar("val_f1_score" if args.phase == 1 else "val_accuracy", train_metric(), i + 1)
        
        if val_metric() > best_metric:
            best_metric = val_metric()
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "best_model.pt",
            )

    writer.close()
