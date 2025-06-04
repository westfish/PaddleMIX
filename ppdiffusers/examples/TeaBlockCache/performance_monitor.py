"""
TeaBlockCache 性能监控器

这个脚本提供了一个简单的装饰器和工具类，可以轻松集成到现有的 TeaBlockCache 实现中
来监控性能瓶颈，无需修改大量代码。

使用方法:
1. 导入 PerformanceMonitor
2. 在 forward 函数中添加关键点的计时
3. 调用 print_analysis() 查看结果

示例:
    from performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    def forward(self, ...):
        monitor.start("total_forward")
        
        # 你的代码
        monitor.start("heuristic_computation")
        # 启发式计算代码
        monitor.end("heuristic_computation")
        
        monitor.end("total_forward")
        
    # 分析结果
    monitor.print_analysis()
"""

import time
import collections
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import numpy as np

class PerformanceMonitor:
    """轻量级性能监控器"""
    
    def __init__(self, enable_detailed_analysis: bool = True):
        """
        初始化性能监控器
        
        Args:
            enable_detailed_analysis: 是否启用详细分析（包括优化建议等）
        """
        self.enable_detailed_analysis = enable_detailed_analysis
        self.reset()
    
    def reset(self):
        """重置所有统计数据"""
        self.timings = collections.defaultdict(list)
        self.current_timers = {}
        self.call_counts = collections.defaultdict(int)
        self.enabled = True
    
    def start(self, name: str):
        """开始计时"""
        if self.enabled:
            self.current_timers[name] = time.perf_counter()
    
    def end(self, name: str) -> float:
        """结束计时并返回耗时"""
        if self.enabled and name in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[name]
            self.timings[name].append(elapsed)
            self.call_counts[name] += 1
            del self.current_timers[name]
            return elapsed
        return 0.0
    
    @contextmanager
    def timer(self, name: str):
        """上下文管理器形式的计时器"""
        self.start(name)
        try:
            yield
        finally:
            self.end(name)
    
    def get_stats(self) -> Tuple[Dict, float]:
        """获取统计信息"""
        stats = {}
        total_time = 0
        
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'count': len(times),
                    'std_time': np.std(times) if len(times) > 1 else 0
                }
                total_time += stats[name]['total_time']
        
        # 计算百分比
        for name in stats:
            stats[name]['percentage'] = (stats[name]['total_time'] / total_time * 100) if total_time > 0 else 0
        
        return stats, total_time
    
    def print_analysis(self, show_details: bool = True):
        """打印性能分析结果"""
        stats, total_time = self.get_stats()
        
        if not stats:
            print("没有收集到性能数据")
            return
        
        print("\n" + "="*80)
        print("TeaBlockCache 性能分析报告")
        print("="*80)
        print(f"总耗时: {total_time:.6f}s")
        print(f"监控组件数: {len(stats)}")
        print("-"*80)
        
        if show_details:
            print(f"{'组件名称':<30} {'总时间(s)':<12} {'平均(s)':<10} {'占比(%)':<8} {'次数':<6} {'标准差':<10}")
            print("-"*80)
            
            # 按总时间排序
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
            
            for name, stat in sorted_stats:
                print(f"{name:<30} {stat['total_time']:<12.6f} {stat['avg_time']:<10.6f} "
                      f"{stat['percentage']:<8.2f} {stat['count']:<6} {stat['std_time']:<10.6f}")
        
        print("-"*80)
        
        # 简要分析
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        if sorted_stats:
            top_consumer = sorted_stats[0]
            print(f"最耗时组件: {top_consumer[0]} ({top_consumer[1]['percentage']:.2f}%)")
            
            # 缓存相关分析
            cache_related = [name for name, _ in sorted_stats 
                           if any(keyword in name.lower() for keyword in ['cache', 'heuristic', 'modulated', 'state'])]
            if cache_related:
                cache_time = sum(stats[name]['total_time'] for name in cache_related)
                cache_percentage = cache_time / total_time * 100
                print(f"缓存相关耗时: {cache_time:.6f}s ({cache_percentage:.2f}%)")
        
        if self.enable_detailed_analysis:
            self._print_optimization_suggestions(stats, total_time)
    
    def _print_optimization_suggestions(self, stats: Dict, total_time: float):
        """打印优化建议"""
        print("\n" + "="*60)
        print("优化建议")
        print("="*60)
        
        suggestions = []
        
        for name, stat in stats.items():
            percentage = stat['percentage']
            avg_time = stat['avg_time']
            count = stat['count']
            
            # 基于时间占比的建议
            if percentage > 20:
                suggestions.append({
                    'priority': 'HIGH',
                    'component': name,
                    'percentage': percentage,
                    'suggestion': self._get_optimization_suggestion(name, stat)
                })
            elif percentage > 10:
                suggestions.append({
                    'priority': 'MEDIUM',
                    'component': name,
                    'percentage': percentage,
                    'suggestion': self._get_optimization_suggestion(name, stat)
                })
            elif percentage > 5 and count > 100:  # 高频调用
                suggestions.append({
                    'priority': 'LOW',
                    'component': name,
                    'percentage': percentage,
                    'suggestion': f"高频调用组件，可优化每次调用的效率"
                })
        
        # 按优先级排序并打印
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions.sort(key=lambda x: (priority_order[x['priority']], -x['percentage']))
        
        if not suggestions:
            print("✅ 当前性能分布相对均衡")
        else:
            for suggestion in suggestions:
                priority_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[suggestion['priority']]
                print(f"\n{priority_icon} {suggestion['priority']} - {suggestion['component']} ({suggestion['percentage']:.1f}%)")
                print(f"   建议: {suggestion['suggestion']}")
    
    def _get_optimization_suggestion(self, name: str, stat: Dict) -> str:
        """根据组件名称和统计数据生成优化建议"""
        name_lower = name.lower()
        
        if 'heuristic' in name_lower:
            return "考虑降低启发式计算频率或简化相似性度量算法"
        elif 'cache' in name_lower and 'operation' in name_lower:
            return "优化缓存操作，减少不必要的 clone() 和状态更新"
        elif 'modulated_input' in name_lower:
            return "考虑复用已计算的归一化结果，避免重复计算"
        elif 'distance' in name_lower:
            return "优化距离计算算法，或使用近似计算方法"
        elif 'state' in name_lower:
            return "优化状态管理，使用更高效的数据结构"
        elif 'block_computation' in name_lower:
            return "这是实际计算时间，如果占比过高可能需要模型级别优化"
        elif 'initialization' in name_lower:
            return "减少初始化开销，考虑延迟初始化或预计算"
        else:
            return "分析具体实现，寻找计算瓶颈"
    
    def save_report(self, filename: str = "performance_report.txt"):
        """保存性能报告到文件"""
        stats, total_time = self.get_stats()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TeaBlockCache 性能分析报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"总耗时: {total_time:.6f}s\n")
            f.write(f"监控组件数: {len(stats)}\n\n")
            
            f.write("详细统计:\n")
            f.write("-"*50 + "\n")
            
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
            for name, stat in sorted_stats:
                f.write(f"{name}:\n")
                f.write(f"  总时间: {stat['total_time']:.6f}s ({stat['percentage']:.2f}%)\n")
                f.write(f"  平均时间: {stat['avg_time']:.6f}s\n")
                f.write(f"  调用次数: {stat['count']}\n")
                f.write(f"  标准差: {stat['std_time']:.6f}s\n\n")
        
        print(f"报告已保存到: {filename}")
    
    def get_cache_overhead_analysis(self) -> Dict:
        """获取缓存开销详细分析"""
        stats, total_time = self.get_stats()
        
        # 分类组件
        categories = {
            'cache_operations': [],
            'heuristic_computation': [],
            'actual_computation': [],
            'initialization': [],
            'others': []
        }
        
        for name, stat in stats.items():
            name_lower = name.lower()
            if 'cache' in name_lower:
                categories['cache_operations'].append((name, stat))
            elif any(keyword in name_lower for keyword in ['heuristic', 'distance', 'modulated']):
                categories['heuristic_computation'].append((name, stat))
            elif any(keyword in name_lower for keyword in ['computation', 'block']):
                categories['actual_computation'].append((name, stat))
            elif 'init' in name_lower:
                categories['initialization'].append((name, stat))
            else:
                categories['others'].append((name, stat))
        
        # 计算各类别统计
        analysis = {}
        for category, items in categories.items():
            if items:
                total_time_cat = sum(stat['total_time'] for _, stat in items)
                analysis[category] = {
                    'total_time': total_time_cat,
                    'percentage': (total_time_cat / total_time * 100) if total_time > 0 else 0,
                    'count': len(items),
                    'components': [name for name, _ in items]
                }
        
        return analysis

# 装饰器形式的简易监控
def profile_function(monitor: PerformanceMonitor, name: str):
    """函数装饰器，用于自动监控函数性能"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor.start(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end(name)
        return wrapper
    return decorator

# 全局监控器实例（可选）
global_monitor = PerformanceMonitor()

def quick_profile(name: str):
    """快速性能监控装饰器，使用全局监控器"""
    return profile_function(global_monitor, name)

# 使用示例
if __name__ == "__main__":
    # 示例使用
    monitor = PerformanceMonitor()
    
    # 模拟一些性能数据
    for i in range(10):
        monitor.start("heuristic_computation")
        time.sleep(0.01)  # 模拟计算
        monitor.end("heuristic_computation")
        
        monitor.start("cache_operations")
        time.sleep(0.005)  # 模拟缓存操作
        monitor.end("cache_operations")
        
        monitor.start("block_computation")
        time.sleep(0.02)  # 模拟实际计算
        monitor.end("block_computation")
    
    # 打印分析结果
    monitor.print_analysis()
    
    # 获取缓存开销分析
    cache_analysis = monitor.get_cache_overhead_analysis()
    print("\n缓存开销分析:")
    for category, data in cache_analysis.items():
        print(f"{category}: {data['percentage']:.2f}% ({data['total_time']:.6f}s)")
    
    # 保存报告
    monitor.save_report("example_performance_report.txt") 