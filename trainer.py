import time
import torch
from pathlib import Path

from utils import (
    rand_fix,
    present_time,
    print_about_train_time,
    save_checkpoint,
    save_train_result,
    print_train_finish,
    save_test_result,
)
from loss import (
    dis_ganLoss,
    gen_ganLoss,
    gen_reconLoss,
)

class Trainer:
    def __init__(
        self,
        gen,
        dis,
        g_optim,
        d_optim,
        dataloader,
        i_val_dataloader,
        p_val_dataloader,
        transform,
        cfg,
        args,
    ):
        self.gen = gen
        self.dis = dis
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.cfg = cfg
        self.loss_cfg = cfg["loss_config"]
        self.main_cfg = cfg["main_config"]
        self.train_cfg = cfg["train_config"]
        self.num_epochs = self.train_cfg["num_epochs"]
        self.args = args
        self.dloader = dataloader
        self.i_val_dataloader = i_val_dataloader
        self.p_val_dataloader = p_val_dataloader
        self.transform = transform
        
    def train(self, epoch):
        device = torch.device(
            f"cuda:{self.args.gpu}" if (torch.cuda.is_available()) else "cpu"
        )
        rand_fix(self.train_cfg["seed"])

        Train_start = time.time()
        print("# Training Start!", end=" ")
        print(present_time())
        print("====================================")
        for epoch in range(epoch + 1, self.num_epochs + 1):
            epoch_start = time.time()
            print(f"Epoch {epoch}/{self.num_epochs}")
            print("-" * 10)

            dis_ganLosses = []
            dis_ganReconLosses = []

            gen_ganLosses = []
            gen_ganReconLosses = []
            gen_reconLosses = []
            
            #pbar = enumerate(self.dloader)
            #pbar = tqdm(pbar, total=len(self.dloader),
            #                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, sample_dict in enumerate(self.dloader):
                iEnc_sample = sample_dict["iEnc_sample"]
                pEnc_sample = sample_dict["pEnc_sample"]
                real_sample = sample_dict["real_sample"]
                recon_sample = sample_dict["recon_sample"]

                self.gen.train()
                self.dis.train()
                self.g_optim.zero_grad(set_to_none=True)
                self.d_optim.zero_grad(set_to_none=True)

                iEnc_sample = iEnc_sample.to(device)
                pEnc_sample = pEnc_sample.to(device)
                real_sample = real_sample.to(device)
                recon_sample = recon_sample.to(device)

                fusion_oups, pose_feats, iden_feats = self.gen(iEnc_sample, pEnc_sample)

                recon_oups, _, _ = self.gen(real_sample, iEnc_sample)

                dis_real_inps = torch.cat([iEnc_sample, real_sample], dim=1)
                dis_fake_inps = torch.cat([fusion_oups, real_sample], dim=1)

                real_logits = self.dis(dis_real_inps)
                fake_logits = self.dis(dis_fake_inps.detach())

                real_samples_cat = torch.cat([iEnc_sample, real_sample], dim=1)
                recon_oups_cat = torch.cat([recon_oups, real_sample], dim=1)
                
                real_recon_logits = self.dis(real_samples_cat)
                fake_recon_logits = self.dis(recon_oups_cat.detach())

                Real_label = torch.full(
                    (real_logits.size()), 1.0, dtype=torch.float, device=device
                )

                RealRecon_label = torch.full(
                    (real_recon_logits.size()), 1.0, dtype=torch.float, device=device
                )

                Fake_label = torch.full(
                    (fake_logits.size()), 0.0, dtype=torch.float, device=device
                )

                FakeRecon_label = torch.full(
                    (fake_recon_logits.size()), 0.0, dtype=torch.float, device=device
                )
                if real_logits.isnan().sum():
                    print("NaN_real")
                    print(real_logits.datach())
                if fake_logits.isnan().sum():
                    print("NaN_fake")
                    print(fake_logits)
                try:
                    _dis_ganLoss, fakeSamples_loss, realSamples_loss = dis_ganLoss(
                        self.loss_cfg, real_logits, Real_label, fake_logits, Fake_label
                    )
                    dis_ganLosses.append(_dis_ganLoss.item())

                    dis_ganReconLoss, _, _ = dis_ganLoss(
                        self.loss_cfg,
                        real_recon_logits,
                        RealRecon_label,
                        fake_recon_logits,
                        FakeRecon_label,
                    )
                    dis_ganReconLosses.append(dis_ganReconLoss.item())
                except:
                    # If get NaN loss according to the difference in the learning degree of gen and dis.
                    pass

                dis_loss = torch.mean(
                    torch.stack([_dis_ganLoss, dis_ganReconLoss], dim=0,)
                )
                dis_loss.backward()
                self.d_optim.step()

                # Train Generator
                fake_logits = self.dis(dis_fake_inps)
                fake_recon_logits = self.dis(recon_oups_cat)

                Real_label = torch.full(
                    (fake_logits.size()), 1.0, dtype=torch.float, device=device
                )
                RealRecon_label = torch.full(
                    (fake_recon_logits.size()), 1.0, dtype=torch.float, device=device
                )

                __gen_ganLoss = gen_ganLoss(self.loss_cfg, fake_logits, Real_label)
                gen_ganLosses.append(__gen_ganLoss.item())

                gen_ganReconLoss = gen_ganLoss(
                    self.loss_cfg, fake_recon_logits, RealRecon_label
                )
                gen_ganReconLosses.append(gen_ganReconLoss.item())

                _gen_ganLoss = torch.mean(
                    torch.stack([__gen_ganLoss, gen_ganReconLoss], dim=0,)
                )

                _gen_reconLoss = gen_reconLoss(self.loss_cfg, recon_oups, iEnc_sample)
                gen_reconLosses.append(_gen_reconLoss.item())

                gen_losses = _gen_ganLoss + _gen_reconLoss
                gen_losses.backward()
                self.g_optim.step()
                if i % 200 == 0:
                    ### Print Loss
                    print(
                        f"[{i}/{len(self.dloader)}]\nD_GANLoss : {_dis_ganLoss.item():.4f}/F:{fakeSamples_loss.item():.4f}/R:{realSamples_loss.item():.4f}    D_GanRecon_Loss: {dis_ganReconLoss.item():.4f}"
                    )
                    print(
                        f"G_GANLoss : {__gen_ganLoss.item():.4f}    G_GanRecon_Loss: {gen_ganReconLoss.item():.4f}    G_Recon_Loss : {_gen_reconLoss.item():.4f}"
                    )
            ### Save the result images
            save_train_result(
                self.gen, iEnc_sample, pEnc_sample, epoch, self.main_cfg,
            )
            save_test_result(
                self.gen,
                self.i_val_dataloader,
                epoch,
                self.main_cfg,
                self.args,
            )
                    
            self.d_optim.step()
            self.g_optim.step()
            print("--------------------------------------------------")
            print(
                f"{epoch}Epoch average Losses\nD_GANLoss : {sum(dis_ganLosses)/len(dis_ganLosses):.4f}   D_GanRecon_Loss: {sum(dis_ganReconLosses)/len(dis_ganReconLosses):.4f}"
            )
            print(
                f"G_GANLoss : {sum(gen_ganLosses)/len(gen_ganLosses):.4f}    G_GanRecon_Loss: {sum(gen_ganReconLosses)/len(gen_ganReconLosses):.4f}    G_Recon_Loss : {sum(gen_reconLosses)/len(gen_reconLosses):.4f}"
            )
            print("--------------------------------------------------")

            dis_ganLosses = []
            dis_ganReconLosses = []

            gen_ganLosses = []
            gen_ganReconLosses = []
            gen_reconLosses = []
                
            save_checkpoint(
                {
                    "epoch": epoch,
                    "gen_state_dict": self.gen.state_dict(),
                    "disc_state_dict": self.dis.state_dict(),
                    "gen_opt": self.g_optim.state_dict(),
                    "disc_opt": self.d_optim.state_dict(),
                },
                Path(f"{self.main_cfg['result_path']}/{self.main_cfg['result_title']}"),
                epoch,
                latest=True
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "gen_state_dict": self.gen.state_dict(),
                    "disc_state_dict": self.dis.state_dict(),
                    "gen_opt": self.g_optim.state_dict(),
                    "disc_opt": self.d_optim.state_dict(),
                },
                Path(f"{self.main_cfg['result_path']}/{self.main_cfg['result_title']}"),
                epoch,
                interval=self.main_cfg.interval,
                minimum_epoch=self.main_cfg.save_minimum_epoch,
            )
            print_about_train_time(epoch_start, epoch, self.train_cfg)

        print_train_finish(self.train_cfg, Train_start)