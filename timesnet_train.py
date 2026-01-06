
from pathlib import Path

from models import TimesNet
import A_dataset
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def diffusion_train(configs):
    #从kdd.norm中读取，划分出train，test
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = TimesNet.Model(configs).to(configs.device)
    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    #当炼丹跑到75%和90%时，把学习率降低10倍
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(configs.epoch_diff):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        model_optim.zero_grad()

        for observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu in tqdm(
            train_loader, desc=f"Train {epoch+1}/{configs.epoch_diff}", leave=False
        ):
            # 在这里把所有需要用到的张量从 CPU 移动到 GPU
            observed_data = observed_data_cpu.to(configs.device)
            observed_mask = observed_mask_cpu.to(configs.device)
            gt_mask = gt_mask_cpu.to(configs.device)
            observed_dataf = observed_dataf_cpu.to(configs.device)

            iter_count += 1
            model_optim.zero_grad()
            x_enc = observed_data * observed_mask
            imputed_output = model(x_enc, None, None, None, mask=observed_mask)
            eval_mask = gt_mask - observed_mask
            loss = loss_fn(imputed_output[eval_mask == 1], observed_data[eval_mask == 1])
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch_diff-1:
            train_loss = np.average(train_loss)
            print("Epoch: {}. Cost time: {}. Train_loss: {}.".format(epoch + 1, time.time() - epoch_time, train_loss))
    return model

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    target_2d = []
    forecast_2d = []
    eval_p_2d = []
    generate_data2d = [] # <--- 我们把它加回来了，用于保存 .csv

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)

    for observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu in tqdm(
        test_loader, desc="Test", leave=False
    ):
        # 同样，在这里把所有需要用到的张量从 CPU 移动到 GPU
        observed_data = observed_data_cpu.to(configs.device)
        observed_mask = observed_mask_cpu.to(configs.device)
        gt_mask = gt_mask_cpu.to(configs.device)
        observed_dataf = observed_dataf_cpu.to(configs.device)

        # --- 核心 ---
        x_enc = observed_data * observed_mask
        with torch.no_grad():
            imputed_output = model(x_enc, None, None, None, mask=observed_mask)
        eval_mask = gt_mask - observed_mask
        
        imputed_sample = imputed_output.detach().to("cpu")
        observed_data = observed_data.detach().to("cpu")
        observed_mask = observed_mask.detach().to("cpu")
        gt_mask = gt_mask.detach().to("cpu")
        #for CRPS
        imputed_data = observed_mask * observed_data + (1 - observed_mask) * imputed_sample
        evalmask = gt_mask - observed_mask

        target_2d.append(observed_data)
        forecast_2d.append(imputed_data)
        eval_p_2d.append(evalmask) 

        B, L, K = imputed_data.shape
        temp = imputed_data.reshape(B*L, K).numpy()
        generate_data2d.append(temp)

    generate_data2d = np.vstack(generate_data2d)
    results_root = Path(getattr(configs, "results_root", Path(".")))
    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / f"TimeNet_{configs.dataset}_p{int(configs.missing_rate*100)}_seed{configs.seed}.csv"
    np.savetxt(out_path, generate_data2d, delimiter=",")
    print(f"TimeNet imputation results saved to {out_path}") # 打印提示

    target_2d = torch.cat(target_2d, dim=0)
    forecast_2d = torch.cat(forecast_2d, dim=0)
    eval_p_2d = torch.cat(eval_p_2d, dim=0)


    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)
    MAE = calc_MAE(target_2d, forecast_2d, eval_p_2d)

    print("RMSE: ", RMSE)
    print("MAE: ", MAE)

def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p])**2)
    return torch.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))
    return error_mean
