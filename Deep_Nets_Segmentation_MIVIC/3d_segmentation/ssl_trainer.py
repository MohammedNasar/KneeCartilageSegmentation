import os
import time
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from utils.ops import aug_rand, rot_rand
from utils.visualize import *


"""Save the model checkpoint."""
def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


"""Training loop for SSL model using ViT architecture."""
def ssl_trainer_vit(train_params, model, train_loader, val_loader, optimizer, 
                recon_loss, contrastive_loss, 
                train_ds, logdir_path, device)-> tuple[torch.nn.Module, int]:
    """Training loop for SSL model."""
    max_epochs = train_params["max_epochs"]
    val_interval = train_params["val_interval"]
    epoch_loss_values = []
    step_loss_values = []
    epoch_cl_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0
    for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            epoch_cl_loss = 0
            epoch_recon_loss = 0
            step = 0

            for batch_data in train_loader:
                step += 1
                start_time = time.time()

                inputs, inputs_2, gt_input = (
                    batch_data["image"].to(device),
                    batch_data["image_2"].to(device),
                    batch_data["gt_image"].to(device),
                )
                optimizer.zero_grad()
                outputs_v1, hidden_v1 = model(inputs)
                outputs_v2, hidden_v2 = model(inputs_2)

                flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
                flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

                r_loss = recon_loss(outputs_v1, gt_input)
                cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

                # Adjust the CL loss by Recon Loss
                total_loss = r_loss + cl_loss * r_loss

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                step_loss_values.append(total_loss.item())

                # CL & Recon Loss Storage of Value
                epoch_cl_loss += cl_loss.item()
                epoch_recon_loss += r_loss.item()

                end_time = time.time()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"train_loss: {total_loss.item():.4f}, "
                    f"time taken: {end_time-start_time}s"
                )

            epoch_loss /= step
            epoch_cl_loss /= step
            epoch_recon_loss /= step

            epoch_loss_values.append(epoch_loss)
            epoch_cl_loss_values.append(epoch_cl_loss)
            epoch_recon_loss_values.append(epoch_recon_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if epoch % val_interval == 0:
                print("Entering Validation for epoch: {}".format(epoch + 1))
                total_val_loss = 0
                val_step = 0
                model.eval()
                for val_batch in val_loader:
                    val_step += 1
                    start_time = time.time()
                    inputs, gt_input = (
                        val_batch["image"].to(device),
                        val_batch["gt_image"].to(device),
                    )
                    print("Input shape: {}".format(inputs.shape))
                    outputs, outputs_v2 = model(inputs)
                    val_loss = recon_loss(outputs, gt_input)
                    total_val_loss += val_loss.item()
                    end_time = time.time()

                total_val_loss /= val_step
                val_loss_values.append(total_val_loss)
                print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

                if total_val_loss < best_val_loss:
                    print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                    best_val_loss = total_val_loss
                    checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(logdir_path, "best_pretrained_model.pt"))

                plt.figure(1, figsize=(8, 8))
                plt.subplot(2, 2, 1)
                plt.plot(epoch_loss_values)
                plt.grid()
                plt.title("Training Loss")

                plt.subplot(2, 2, 2)
                plt.plot(val_loss_values)
                plt.grid()
                plt.title("Validation Loss")

                plt.subplot(2, 2, 3)
                plt.plot(epoch_cl_loss_values)
                plt.grid()
                plt.title("Training Contrastive Loss")

                plt.subplot(2, 2, 4)
                plt.plot(epoch_recon_loss_values)
                plt.grid()
                plt.title("Training Recon Loss")

                plt.savefig(os.path.join(logdir_path, "loss_plots.png"))
                plt.close(1)
    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    return model, epoch


"""Validate the model using the validation set."""
def ssl_validator_swin(args, model, loss_function, test_loader):
    model.eval()
    loss_val = []
    loss_val_recon = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            val_inputs = batch["image"].cuda()
            x1, rot1 = rot_rand(args, val_inputs)
            x2, rot2 = rot_rand(args, val_inputs)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            with autocast(enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
            loss_recon = losses_tasks[2]
            loss_val.append(loss.item())
            loss_val_recon.append(loss_recon.item())
            x_gt = x1.detach().cpu().numpy()
            x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
            xgt = x_gt[0][0][:, :, 48] * 255.0
            xgt = xgt.astype(np.uint8)
            x1_augment = x1_augment.detach().cpu().numpy()
            x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
            x_aug = x1_augment[0][0][:, :, 48] * 255.0
            x_aug = x_aug.astype(np.uint8)
            rec_x1 = rec_x1.detach().cpu().numpy()
            rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
            recon = rec_x1[0][0][:, :, 48] * 255.0
            recon = recon.astype(np.uint8)
            img_list = [xgt, x_aug, recon]
            print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_recon))

    return np.mean(loss_val), np.mean(loss_val_recon), img_list


"""Training loop for SSL model using Swin Transformer architecture."""
def ssl_trainer_swin(args, model, global_step, train_loader, val_best, scaler,
                     optimizer, scheduler, loss_function, test_loader, logdir, 
                     writer)->tuple[torch.nn.Module, int, float, float]:
    model.train()
    loss_train = []
    loss_train_recon = []

    for step, batch in enumerate(train_loader):
        t1 = time()
        x = batch["image"].cuda()
        x1, rot1 = rot_rand(args, x)
        x2, rot2 = rot_rand(args, x)
        x1_augment = aug_rand(args, x1)
        x2_augment = aug_rand(args, x2)
        x1_augment = x1_augment
        x2_augment = x2_augment
        # display_multiple_slices(x1)
        # display_multiple_slices(x2)
        # breakpoint()
        with autocast(enabled=args.amp):
            rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
            rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
            rot_p = torch.cat([rot1_p, rot2_p], dim=0)
            rots = torch.cat([rot1, rot2], dim=0)
            imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
            imgs = torch.cat([x1, x2], dim=0)
            loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
        loss_train.append(loss.item())
        loss_train_recon.append(losses_tasks[2].item())
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.lrdecay:
            scheduler.step()
        optimizer.zero_grad()
        print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))

        global_step += 1
        val_cond = global_step % args.eval_num == 0

        if val_cond:
            val_loss, val_loss_recon, img_list = ssl_validator_swin(args, model, loss_function, test_loader)
            writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
            writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
            writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

            writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats="HW")
            writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats="HW")
            writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats="HW")

            if val_loss_recon < val_best:
                val_best = val_loss_recon
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                print(
                    "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                        val_best, val_loss_recon
                    )
                )
            else:
                print(
                    "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                        val_best, val_loss_recon
                    )
                )
    return model, global_step, loss, val_best
