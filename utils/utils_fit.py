import os
import sys
import time
import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils import get_lr
from utils.utils_metrics import f_score


def calculate_fps(model, device, input_shape, batch_size=1, test_iterations=100):
    """
    独立的FPS测试函数 - 只在训练完成后调用
    """
    print("🚀 开始FPS性能测试...")
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(batch_size, 3, input_shape[0], input_shape[1]).to(device)
    
    # 预热GPU
    print("⏳ GPU预热中...")
    with torch.no_grad():
        for _ in range(20):  # 增加预热次数
            _ = model(test_input)
    
    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"📊 开始{test_iterations}次推理测试...")
    # 开始正式测试
    time_list = []
    with torch.no_grad():
        for i in range(test_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            time_list.append(end_time - start_time)
            
            # 显示进度
            if (i + 1) % 20 == 0:
                print(f"  完成 {i + 1}/{test_iterations} 次测试...")
    
    # 计算FPS统计
    time_array = np.array(time_list)
    avg_time = np.mean(time_array)
    min_time = np.min(time_array)
    max_time = np.max(time_array)
    std_time = np.std(time_array)
    
    fps_avg = batch_size / avg_time
    fps_max = batch_size / min_time
    fps_min = batch_size / max_time
    
    print(f"⚡ FPS测试结果:")
    print(f"   平均FPS: {fps_avg:.2f}")
    print(f"   最大FPS: {fps_max:.2f}")
    print(f"   最小FPS: {fps_min:.2f}")
    print(f"   标准差: {std_time*1000:.2f}ms")
    print(f"   平均推理时间: {avg_time*1000:.2f}ms")
    
    model.train()  # 恢复训练模式
    return fps_avg, fps_max, fps_min, avg_time

def test_model_fps(model, device, input_shape, log_dir, model_name="model", test_iterations=100):
    """
    独立的FPS测试函数，结果保存到指定的log目录
    Args:
        model: 训练好的模型
        device: 计算设备
        input_shape: 输入图像尺寸 [H, W]
        log_dir: 日志保存目录
        model_name: 模型名称
        test_iterations: 测试迭代次数
    """
    print("\n" + "="*60)
    print("🚀 开始模型FPS性能测试")
    print("="*60)
    
    model.eval()
    
    # 测试不同batch_size
    batch_sizes = [1, 2, 4, 8]
    fps_results = {}
    test_results = []
    
    for batch_size in batch_sizes:
        print(f"\n🧪 测试 Batch Size = {batch_size}")
        
        try:
            # 创建测试输入
            test_input = torch.randn(batch_size, 3, input_shape[0], input_shape[1]).to(device)
            
            # 预热GPU
            print("⏳ GPU预热中...")
            with torch.no_grad():
                for _ in range(20):
                    _ = model(test_input)
            
            # 同步GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
                # 检查显存使用
                memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024   # MB
                print(f"📊 显存使用: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB")
            
            print(f"🚀 开始{test_iterations}次推理测试...")
            time_list = []
            
            # 进行FPS测试
            with torch.no_grad():
                for i in range(test_iterations):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    _ = model(test_input)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    time_list.append(end_time - start_time)
                    
                    # 显示进度
                    if (i + 1) % (test_iterations // 4) == 0:
                        print(f"  进度: {i + 1}/{test_iterations}")
            
            # 计算统计数据
            time_array = np.array(time_list)
            avg_time = np.mean(time_array)
            min_time = np.min(time_array)
            max_time = np.max(time_array)
            std_time = np.std(time_array)
            
            fps_avg = batch_size / avg_time
            fps_max = batch_size / min_time
            fps_min = batch_size / max_time
            
            result = {
                'batch_size': batch_size,
                'fps_avg': fps_avg,
                'fps_max': fps_max,
                'fps_min': fps_min,
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'std_time_ms': std_time * 1000,
                'memory_mb': memory_allocated if device.type == 'cuda' else 0
            }
            
            fps_results[batch_size] = result
            test_results.append(result)
            
            print(f"✅ 结果:")
            print(f"   平均FPS: {fps_avg:.2f}")
            print(f"   最大FPS: {fps_max:.2f}")
            print(f"   最小FPS: {fps_min:.2f}")
            print(f"   平均推理时间: {avg_time*1000:.2f}ms")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch Size {batch_size} 显存不足: {str(e)}")
            else:
                print(f"❌ Batch Size {batch_size} 测试失败: {str(e)}")
            fps_results[batch_size] = None
    
    # 保存结果到log目录
    save_fps_results(test_results, log_dir, model_name, input_shape, device, test_iterations)
    
    model.train()  # 恢复训练模式
    return fps_results

def save_fps_results(test_results, log_dir, model_name, input_shape, device, test_iterations):
    """保存FPS测试结果到log目录"""
    
    # 确保log目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存详细的文本报告
    fps_report_path = os.path.join(log_dir, f"fps_test_report_{timestamp}.txt")
    with open(fps_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("FPS性能测试报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型名称: {model_name}\n")
        f.write(f"输入尺寸: {input_shape}\n")
        f.write(f"计算设备: {device}\n")
        f.write(f"测试次数: {test_iterations}\n\n")
        
        f.write("测试结果:\n")
        f.write("-" * 40 + "\n")
        
        for result in test_results:
            f.write(f"\nBatch Size {result['batch_size']}:\n")
            f.write(f"  平均FPS: {result['fps_avg']:.2f}\n")
            f.write(f"  最大FPS: {result['fps_max']:.2f}\n")
            f.write(f"  最小FPS: {result['fps_min']:.2f}\n")
            f.write(f"  平均推理时间: {result['avg_time_ms']:.2f}ms\n")
            f.write(f"  最小推理时间: {result['min_time_ms']:.2f}ms\n")
            f.write(f"  最大推理时间: {result['max_time_ms']:.2f}ms\n")
            f.write(f"  时间标准差: {result['std_time_ms']:.2f}ms\n")
            if result['memory_mb'] > 0:
                f.write(f"  显存使用: {result['memory_mb']:.1f}MB\n")
        
        # 推荐配置
        if test_results:
            best_result = max(test_results, key=lambda x: x['fps_avg'])
            f.write(f"\n推荐配置:\n")
            f.write(f"  最佳Batch Size: {best_result['batch_size']}\n")
            f.write(f"  最佳FPS: {best_result['fps_avg']:.2f}\n")
    
    # 2. 保存简单的CSV格式数据
    csv_path = os.path.join(log_dir, "fps_results.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Batch_Size,Avg_FPS,Max_FPS,Min_FPS,Avg_Time_ms,Memory_MB\n")
        for result in test_results:
            f.write(f"{result['batch_size']},{result['fps_avg']:.2f},{result['fps_max']:.2f},"
                   f"{result['fps_min']:.2f},{result['avg_time_ms']:.2f},{result['memory_mb']:.1f}\n")
    
    # 3. 绘制FPS图表
    plot_fps_charts(test_results, log_dir)
    
    print(f"\n📁 FPS测试结果已保存:")
    print(f"   详细报告: {fps_report_path}")
    print(f"   CSV数据: {csv_path}")
    print(f"   图表: {os.path.join(log_dir, 'fps_chart.png')}")

def plot_fps_charts(test_results, log_dir):
    """绘制FPS性能图表"""
    if not test_results:
        return
    
    # 准备数据
    batch_sizes = [r['batch_size'] for r in test_results]
    fps_avg = [r['fps_avg'] for r in test_results]
    fps_max = [r['fps_max'] for r in test_results]
    fps_min = [r['fps_min'] for r in test_results]
    avg_times = [r['avg_time_ms'] for r in test_results]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. FPS对比图
    ax1.bar(batch_sizes, fps_avg, alpha=0.7, color='skyblue', label='Average FPS')
    ax1.plot(batch_sizes, fps_max, 'ro-', label='Max FPS', markersize=6)
    ax1.plot(batch_sizes, fps_min, 'go-', label='Min FPS', markersize=6)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('FPS')
    ax1.set_title('FPS Performance by Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(fps_avg):
        ax1.text(batch_sizes[i], v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    # 2. 推理时间图
    ax2.plot(batch_sizes, avg_times, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Average Inference Time (ms)')
    ax2.set_title('Inference Time by Batch Size')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(avg_times):
        ax2.text(batch_sizes[i], v, f'{v:.1f}ms', ha='center', va='bottom')
    
    # 3. 吞吐量对比（总FPS）
    total_fps = [r['fps_avg'] for r in test_results]
    ax3.bar(batch_sizes, total_fps, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Throughput (Images/Second)')
    ax3.set_title('Model Throughput')
    ax3.grid(True, alpha=0.3)
    
    # 4. 显存使用情况
    if any(r['memory_mb'] > 0 for r in test_results):
        memory_usage = [r['memory_mb'] for r in test_results]
        ax4.plot(batch_sizes, memory_usage, 'ro-', linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('GPU Memory Usage')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'GPU Memory\nData Not Available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(log_dir, 'fps_performance_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建简单的FPS对比图
    plt.figure(figsize=(10, 6))
    plt.bar(batch_sizes, fps_avg, alpha=0.7, color='skyblue', label='Average FPS')
    plt.plot(batch_sizes, fps_max, 'ro-', label='Max FPS', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('FPS (Frames Per Second)')
    plt.title('Model FPS Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(fps_avg):
        plt.text(batch_sizes[i], v + max(fps_avg)*0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    simple_chart_path = os.path.join(log_dir, 'fps_chart.png')
    plt.savefig(simple_chart_path, dpi=300, bbox_inches='tight')
    plt.close()

# 保持原有的fit_one_epoch函数，但移除FPS计算
def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank):
    total_loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                              'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice
        val_loss += loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth"))

        # 移除fps参数
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('')
        
        # 只返回loss，不返回fps
        return total_loss / epoch_step, val_loss / epoch_step_val
    else:
        return None, None