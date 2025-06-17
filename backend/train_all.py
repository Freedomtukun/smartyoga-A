#!/usr/bin/env python3
"""
统一模型训练脚本 - Train All Models

该脚本用于统一训练所有瑜伽姿势相关的模型：
1. 评分模型（Score Model）
2. 分类模型（Classification Model）
3. 视频模型（Video Model - 预留）

Usage:
    python train_all.py
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# 设置日志目录
LOG_DIR = Path("logs/train_all")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 导入评分模型训练函数
try:
    from train_model import train_from_dataset
except ImportError:
    print("⚠️ 警告：无法导入 train_model 模块")
    train_from_dataset = None


class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


# 设置双重输出
tee = TeeOutput(LOG_FILE)
sys.stdout = tee


def train_classifier_model(input_dir="dataset/classify", 
                         output_model="models/classify/classify_model.h5",
                         output_labels="models/classify/class_labels.txt",
                         epochs=15, batch_size=32):
    """
    调用分类模型训练脚本
    
    由于 train_classifier.py 没有导出函数，使用 subprocess 调用
    
    注意：train_classifier.py 当前可能不支持命令行参数。
    如需参数生效，请在 train_classifier.py 中添加 argparse 支持。
    """
    print(f"📂 输入目录: {input_dir}")
    print(f"💾 输出模型: {output_model}")
    print(f"🏷️ 标签文件: {output_labels}")
    print(f"🔄 训练轮数: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    # 构建命令，传递参数
    # 注：如果 train_classifier.py 不支持这些参数，它们会被忽略
    cmd = [
        sys.executable, "scripts/train_classifier.py",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 分类模型训练完成！")
        print(f"   模型已保存至: {output_model}")
        print(f"   标签已保存至: {output_labels}")
        if result.stdout:
            print("📝 训练输出:")
            print(result.stdout)
    else:
        raise Exception(f"分类模型训练失败: {result.stderr}")


def train_sequence_model():
    """
    TODO: 视频序列模型训练函数
    
    预留接口，未来实现 LSTM/Transformer 等序列模型训练
    """
    print("🎬 视频模型训练功能已预留（未启用）")
    print("   TODO: 实现基于视频序列的姿势评分模型")
    print("   - 输入: 视频帧序列")
    print("   - 输出: 时序评分模型")
    print("   - 技术栈: LSTM / Transformer / 3D-CNN")


def train_all():
    """
    统一训练所有模型的主函数
    """
    print("=" * 60)
    print("🚀 开始统一模型训练流程")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 日志文件: {LOG_FILE}")
    print("=" * 60)
    
    # 记录成功和失败的任务
    results = {
        "success": [],
        "failed": []
    }
    
    try:
        # 1. 评分模型训练
        print("\n" + "="*60)
        print("🧘 开始评分模型训练...")
        print("="*60)
        
        try:
            if train_from_dataset is None:
                raise ImportError("train_from_dataset 函数不可用")
            
            # 确保输出目录存在
            os.makedirs("models/score", exist_ok=True)
            
            # 调用评分模型训练
            train_from_dataset(
                dataset_dir="dataset/train",
                model_out="models/score/score_model.h5",
                epochs=2,  # 可以根据需要调整
                batch_size=16,
                workers=4,
                learning_rate=0.001,
                validation_split=0.1,
                use_multi_head=False,
                mode='image_classification',
                plots_dir="training_plots/score"
            )
            
            print("✅ 评分模型训练成功！")
            results["success"].append("评分模型")
            
        except Exception as e:
            print(f"❌ 评分模型训练失败: {str(e)}")
            results["failed"].append(("评分模型", str(e)))
        
        # 2. 分类模型训练
        print("\n" + "="*60)
        print("🧠 开始分类模型训练...")
        print("="*60)
        
        try:
            train_classifier_model(
                input_dir="dataset/classify",
                output_model="models/classify/classify_model.h5",
                output_labels="models/classify/class_labels.txt",
                epochs=15,
                batch_size=32
            )
            results["success"].append("分类模型")
            
        except Exception as e:
            print(f"❌ 分类模型训练失败: {str(e)}")
            results["failed"].append(("分类模型", str(e)))
        
        # 3. 视频模型训练（预留）
        print("\n" + "="*60)
        try:
            train_sequence_model()
            # 由于是预留功能，不计入结果
        except Exception as e:
            print(f"❌ 视频模型功能错误: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被手动中断！")
        print("=" * 60)
        sys.stdout = tee.terminal  # 恢复标准输出
        tee.close()
        sys.exit(130)  # 标准的 KeyboardInterrupt 退出码
    
    # 打印最终结果总结
    print("\n" + "="*60)
    print("📊 训练结果总结")
    print("="*60)
    
    if results["success"]:
        print(f"✅ 成功训练的模型 ({len(results['success'])}个):")
        for model in results["success"]:
            print(f"   - {model}")
    
    if results["failed"]:
        print(f"\n❌ 训练失败的模型 ({len(results['failed'])}个):")
        for model, error in results["failed"]:
            print(f"   - {model}: {error}")
    
    print("\n" + "="*60)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not results["failed"]:
        print("🎉 ✅ 所有训练任务完成！")
    else:
        print(f"⚠️ 部分任务完成！成功: {len(results['success'])}, 失败: {len(results['failed'])}")
    print("="*60)
    
    # 返回是否全部成功
    return len(results["failed"]) == 0


def main():
    """
    脚本入口点
    """
    try:
        # 检查数据准备情况
        print("🔍 检查训练数据准备情况...")
        
        # 检查必要的目录
        required_dirs = ["dataset/train", "dataset/classify"]
        missing_dirs = []
        total_imgs = 0
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
            else:
                # 优化：使用生成器统计，避免创建列表
                count = sum(1 for p in Path(dir_path).rglob("*") 
                           if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
                print(f"✅ {dir_path}: {count} 张图片")
                total_imgs += count
        
        if not missing_dirs:
            print(f"📊 训练总图片数: {total_imgs}")
        
        if missing_dirs:
            print(f"\n❌ 缺少必要的数据目录: {', '.join(missing_dirs)}")
            print("💡 提示：请先运行 prepare_training_data.py 准备训练数据")
            
            # 询问是否自动准备数据
            response = input("\n是否自动准备训练数据？(y/N): ").strip().lower()
            if response == 'y':
                print("\n🚀 正在准备训练数据...")
                result = subprocess.run([sys.executable, "prepare_training_data.py"], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ 数据准备失败: {result.stderr}")
                    return 1
                else:
                    print("✅ 数据准备完成！")
                    if result.stdout:
                        print("📝 准备过程输出:")
                        print(result.stdout)
                    
                    # 重新统计图片数量
                    print("\n📊 重新统计训练数据...")
                    total_imgs = 0
                    for dir_path in required_dirs:
                        count = sum(1 for p in Path(dir_path).rglob("*") 
                                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
                        print(f"✅ {dir_path}: {count} 张图片")
                        total_imgs += count
                    print(f"📊 训练总图片数: {total_imgs}")
            else:
                return 1
        
        # 开始训练
        success = train_all()
        
        # 返回适当的退出码
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被手动中断！")
        return 130
    
    finally:
        # 恢复标准输出并关闭日志文件
        if sys.stdout != sys.__stdout__:
            sys.stdout = tee.terminal
            tee.close()
            print(f"\n📁 完整日志已保存至: {LOG_FILE}")


if __name__ == "__main__":
    sys.exit(main())