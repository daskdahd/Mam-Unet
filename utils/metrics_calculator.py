import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm

class ComprehensiveMetricsCalculator:
    """完整的评估指标计算器：FPS, aACC, mACC"""
    
    def __init__(self, model, device, input_shape=(512, 512), num_classes=21):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def calculate_fps(self, test_samples=100, warmup_samples=10):
        """计算FPS (每秒帧数)"""
        print("🚀 开始计算FPS...")
        
        self.model.eval()
         
        # 创建测试输入
        dummy_input = torch.randn(1, 3, *self.input_shape).to(self.device)
        
        # 预热GPU
        with torch.no_grad():
            for _ in range(warmup_samples):
                _ = self.model(dummy_input)
        
        # 同步GPU（如果使用CUDA）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 开始计时
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(test_samples):
                output = self.model(dummy_input)
        
        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 计算FPS
        total_time = end_time - start_time
        fps = test_samples / total_time
        avg_inference_time = (total_time / test_samples) * 1000  # ms
        
        print(f"✅ FPS计算完成:")
        print(f"   测试样本数: {test_samples}")
        print(f"   总耗时: {total_time:.4f}s")
        print(f"   平均推理时间: {avg_inference_time:.2f}ms")
        print(f"   FPS: {fps:.2f} frames/sec")
        
        return fps, avg_inference_time
    
    def calculate_accuracy_metrics(self, dataloader, max_samples=None):
        """计算aACC和mACC"""
        print("📊 开始计算准确率指标...")
        
        self.model.eval()
        
        # 初始化计数器
        total_correct_pixels = 0
        total_pixels = 0
        class_correct = np.zeros(self.num_classes, dtype=np.int64)
        class_total = np.zeros(self.num_classes, dtype=np.int64)
        
        processed_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="计算准确率")
            for batch_idx, batch in enumerate(pbar):
                # 如果设置了最大样本数限制
                if max_samples and processed_samples >= max_samples:
                    break
                
                # 🔥 修改这里：安全解包数据
                try:
                    if len(batch) == 2:
                        images, labels = batch
                    elif len(batch) == 3:
                        images, labels, _ = batch  # 可能有额外的信息
                    elif len(batch) > 3:
                        images, labels = batch[0], batch[1]  # 只取前两个
                    else:
                        print(f"⚠️ 意外的batch格式，长度: {len(batch)}")
                        continue
                except Exception as e:
                    print(f"⚠️ 数据解包错误: {e}")
                    continue
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 模型预测
                outputs = self.model(images)
                
                # 如果输出尺寸与标签不匹配，进行插值
                if outputs.shape[2:] != labels.shape[1:]:
                    outputs = F.interpolate(
                        outputs, 
                        size=labels.shape[1:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                
                # 获取预测类别
                predictions = torch.argmax(outputs, dim=1)
                
                # 计算整体像素准确率 (aACC)
                correct_pixels = (predictions == labels).sum().item()
                total_correct_pixels += correct_pixels
                total_pixels += labels.numel()
                
                # 计算各类别准确率 (用于mACC)
                for class_id in range(self.num_classes):
                    # 找到真实标签为当前类别的像素
                    class_mask = (labels == class_id)
                    
                    if class_mask.sum() > 0:  # 如果该类别在当前batch中存在
                        # 该类别的正确预测数量
                        class_correct_pred = (predictions[class_mask] == class_id).sum().item()
                        class_correct[class_id] += class_correct_pred
                        class_total[class_id] += class_mask.sum().item()
            
                processed_samples += images.shape[0]
                
                # 更新进度条
                current_aacc = (total_correct_pixels / total_pixels) * 100
                pbar.set_postfix({
                    'aACC': f'{current_aacc:.2f}%',
                    'Samples': processed_samples
                })
        
        # 计算最终的aACC
        aacc = (total_correct_pixels / total_pixels) * 100
        
        # 计算mACC
        class_accuracies = []
        for class_id in range(self.num_classes):
            if class_total[class_id] > 0:
                class_acc = class_correct[class_id] / class_total[class_id]
                class_accuracies.append(class_acc)
            else:
                # 如果某个类别在验证集中不存在，不计入mACC计算
                print(f"⚠️ 警告: 类别 {class_id} 在验证集中未出现")
        
        macc = np.mean(class_accuracies) * 100 if class_accuracies else 0
        
        print(f"✅ 准确率计算完成:")
        print(f"   处理样本数: {processed_samples}")
        print(f"   总像素数: {total_pixels:,}")
        print(f"   正确像素数: {total_correct_pixels:,}")
        print(f"   aACC (整体像素准确率): {aacc:.2f}%")
        print(f"   有效类别数: {len(class_accuracies)}/{self.num_classes}")
        print(f"   mACC (平均类别准确率): {macc:.2f}%")
        
        # 返回详细结果
        return {
            'aacc': aacc,
            'macc': macc,
            'class_accuracies': class_accuracies,
            'total_pixels': total_pixels,
            'correct_pixels': total_correct_pixels,
            'processed_samples': processed_samples
        }
    
    def calculate_all_metrics(self, dataloader, max_samples=None):
        """计算所有指标：FPS + aACC + mACC"""
        print("\n" + "="*60)
        print("🎯 开始完整性能评估...")
        
        # 1. 计算FPS
        fps, avg_time = self.calculate_fps()
        
        # 2. 计算准确率指标
        accuracy_results = self.calculate_accuracy_metrics(dataloader, max_samples)
        
        # 整合所有结果
        all_metrics = {
            'fps': fps,
            'avg_inference_time_ms': avg_time,
            'aacc': accuracy_results['aacc'],
            'macc': accuracy_results['macc'],
            'total_pixels': accuracy_results['total_pixels'],
            'correct_pixels': accuracy_results['correct_pixels'],
            'processed_samples': accuracy_results['processed_samples']
        }
        
        return all_metrics
    
    def save_metrics_report(self, metrics, save_dir, model_name="UNet"):
        """保存详细的评估报告"""
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存JSON格式
        import json
        json_path = os.path.join(save_dir, "final_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存详细报告
        report_path = os.path.join(save_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{model_name} 模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 性能指标:\n")
            f.write(f"🚀 FPS (推理速度): {metrics['fps']:.2f} frames/sec\n")
            f.write(f"⏱️ 平均推理时间: {metrics['avg_inference_time_ms']:.2f} ms/frame\n")
            f.write(f"🎯 aACC (整体像素准确率): {metrics['aacc']:.2f}%\n")
            f.write(f"📈 mACC (平均类别准确率): {metrics['macc']:.2f}%\n\n")
            
            f.write("📋 详细统计:\n")
            f.write(f"- 处理样本数: {metrics['processed_samples']}\n")
            f.write(f"- 总像素数: {metrics['total_pixels']:,}\n")
            f.write(f"- 正确像素数: {metrics['correct_pixels']:,}\n\n")
            
            f.write("📝 指标说明:\n")
            f.write("- FPS: 每秒能处理的图片帧数，越高越好\n")
            f.write("- aACC: 所有像素的分类准确率，反映整体性能\n")
            f.write("- mACC: 各类别准确率的平均值，反映类别平衡性能\n")
        
        print(f"📁 评估报告已保存:")
        print(f"   JSON格式: {json_path}")
        print(f"   详细报告: {report_path}")
        
        return json_path, report_path