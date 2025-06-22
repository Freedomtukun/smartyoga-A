#!/usr/bin/env python3
"""
Unified Training Trigger
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
• 自动判定是否满足训练条件  
• 先训练评分模型，再训练分类模型  
• 训练完成把最新 .h5 复制为 latest_model.h5 方便调用
"""

import os, sys, subprocess, psutil, time, shutil
from datetime import datetime
from pathlib import Path

# ============= 全局配置 =============
TRAIN_IMG_DIR      = "dataset/train"           # 评分模型临时训练集
CLASSIFY_IMG_DIR   = "dataset/image_pool"      # ★ 修改：分类直接复用统一图池
POOL_DIR           = "dataset/image_pool"      # 采样池
LOG_DIR            = "logs"

PREPARE_SCRIPT     = "scripts/prepare_training_data.py"
CSV_SCRIPT         = "cos_tools/generate_image_list.py"
CLASSIFY_SCRIPT    = "scripts/train_classifier.py"

MIN_IMAGES   = 1000           # 至少 N 张图片才启动训练
MAX_IMAGES   = 2600           # 评分/分类均使用的采样上限
MAX_RETRY    = 3              # 评分模型失败重试次数
# =====================================

now          = datetime.now()
NOW_STR      = now.strftime('%Y-%m-%d %H:%M:%S')
log_path     = f"{LOG_DIR}/train_trigger_{now.strftime('%Y%m%d')}.log"
fail_log_path= f"{LOG_DIR}/train_trigger_failed_{now.strftime('%Y%m%d')}.log"
os.makedirs(LOG_DIR, exist_ok=True)

def count_images(folder:str)->int:
    return sum(1 for r,_,fs in os.walk(folder)
                 for f in fs if f.lower().endswith(('.jpg','.jpeg','.png')))

def log(msg:str, fp):
    print(msg)                # 同步到终端
    fp.write(msg + '\n')
    fp.flush()

def check_mem_ok()->bool:
    m = psutil.virtual_memory()
    return m.available / m.total > 0.30     # >30% 空闲

def safe_run(cmd:list|str, fp, **kw):
    res = subprocess.run(cmd, shell=isinstance(cmd,str),
                         capture_output=True, text=True, **kw)
    fp.write(res.stdout); fp.write(res.stderr)
    return res.returncode==0

def sync_latest(model_dir:str, prefix:str, latest_name:str, fp):
    md = Path(model_dir)
    files = list(md.glob(f"{prefix}_model_*.h5"))+list(md.glob(f"{prefix}_model.h5"))
    if not files:
        log(f"⚠️  [sync] 未找到 {prefix} 模型", fp); return
    latest = max(files, key=lambda p:p.stat().st_mtime)
    (md/latest_name).write_bytes(latest.read_bytes())
    log(f"🆕 [sync] {latest.name} → {latest_name}", fp)

with open(log_path,"a",encoding="utf-8") as fp:
    log(f"\n===== {NOW_STR} 触发训练 =====", fp)

    # 0. 数据准备 & CSV
    safe_run(["python3", PREPARE_SCRIPT], fp)
    if not safe_run(["python3", CSV_SCRIPT], fp):
        log("❌ CSV 生成失败——终止", fp); sys.exit(1)

    # --------- 评分模型 ---------
    train_cnt = count_images(TRAIN_IMG_DIR)
    log(f"评分图片数：{train_cnt}", fp)
    if train_cnt < MIN_IMAGES:
        log("图片不足，跳过评分模型训练", fp)
    else:
        if not check_mem_ok():
            log("内存不足，跳过评分模型训练", fp)
        else:
            retry = 0
            while retry < MAX_RETRY:
                cmd = ("python3 train_model.py "
                       f"--epochs 3 --batch-size 16 --max-images {MAX_IMAGES}")
                ok = safe_run(cmd, fp)
                if ok:
                    log("✅ 评分模型训练成功", fp)
                    sync_latest("models/score", "score", "latest_model.h5", fp)
                    break
                retry += 1
                log(f"❌ 评分模型第 {retry} 次失败", fp)
            else:
                log("❌ 评分模型多次失败，放弃", fp)
                Path(fail_log_path).write_text(f"{NOW_STR} 评分模型失败\n")

    # --------- 分类模型 ---------
    cls_cnt = count_images(CLASSIFY_IMG_DIR)
    log(f"分类图片数：{cls_cnt}", fp)
    if cls_cnt < MIN_IMAGES:
        log("分类图片不足，跳过", fp)
    else:
        cmd = ["python3", CLASSIFY_SCRIPT,
               "--epochs","10","--batch-size","8",
               "--max-images", str(MAX_IMAGES)]
        if safe_run(cmd, fp):
            log("✅ 分类模型训练成功", fp)
            sync_latest("models/classify", "classify", "latest_model.h5", fp)
        else:
            log("❌ 分类模型训练失败", fp)

    log(f"===== 训练流程结束 {datetime.now():%H:%M:%S} =====", fp)
