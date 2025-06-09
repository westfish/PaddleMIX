# export FLAGS_sdpa_select_math=yes 

#╔════════════════════════════════════════════════════════════════════════════╗
#║                          📋 脚本使用说明                                   ║
#║                                                                            ║
#║  🆕 新增区域: 包含6组极限优化配置，目标最大化加速比                          ║
#║  📜 原有区域: 完全保留原脚本的所有配置                                      ║
#║                                                                            ║
#║  🚀 快速测试: 运行 ./gen_quick.sh 选择单个配置测试                          ║
#║  📊 批量测试: 运行 ./gen.sh 执行所有配置                                    ║
#║                                                                            ║
#║  🎯 推荐顺序: ULTRA AGGRESSIVE → EXPERIMENTAL EXTREME → VERY AGGRESSIVE   ║
#╚════════════════════════════════════════════════════════════════════════════╝

#╔════════════════════════════════════════════════════════════════════════════╗
#║                     🚀 新增极限加速配置区域 (NEW)                          ║
#║               这些是为了达到最大加速比而新增的优化配置                        ║
#║               配置按激进程度排序，从最激进到平衡配置                         ║
#╚════════════════════════════════════════════════════════════════════════════╝

#=============================================================================
# 🚀 MAXIMUM ACCELERATION CONFIGURATIONS - 新增优化配置
# 以下配置按激进程度排序，目标是达到最大加速比
# 注意：这些是在原有配置基础上新增的极限优化配置
#=============================================================================

echo "🚀 Starting TeaBlockCache Taylor Maximum Acceleration Tests..."
echo "🆕 运行新增的极限优化配置..."

#=============================================================================
# 🔥 ULTRA AGGRESSIVE CONFIGURATION - 最激进配置 (最大加速)
#=============================================================================

echo "🔥 Running ULTRA AGGRESSIVE configuration..."
CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
--step_start 50 \
--step_end 950 \
--block_cache_start 1 \
--single_block_cache_start 1 \
--block_rel_l1_thresh 0.05 \
--single_block_rel_l1_thresh 0.05 \
--rel_l1_thresh 0.1 \
--inference_step 50 \
--taylor_max_order 1 \
--taylor_first_enhance 1 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_ultra_aggressive.log 2>&1 &

#=============================================================================
# ⚡ VERY AGGRESSIVE CONFIGURATION - 次激进配置
#=============================================================================

echo "⚡ Running VERY AGGRESSIVE configuration..."
CUDA_VISIBLE_DEVICES=2 nohup python teablock_generation.py \
--step_start 80 \
--step_end 920 \
--block_cache_start 2 \
--single_block_cache_start 2 \
--block_rel_l1_thresh 0.1 \
--single_block_rel_l1_thresh 0.1 \
--rel_l1_thresh 0.15 \
--inference_step 50 \
--taylor_max_order 2 \
--taylor_first_enhance 1 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_very_aggressive.log 2>&1 &

#=============================================================================
# 🎯 TAYLOR OPTIMIZED CONFIGURATION - Taylor专用优化
#=============================================================================

echo "🎯 Running TAYLOR OPTIMIZED configuration..."
CUDA_VISIBLE_DEVICES=3 nohup python teablock_generation.py \
--step_start 100 \
--step_end 900 \
--block_cache_start 3 \
--single_block_cache_start 3 \
--block_rel_l1_thresh 0.2 \
--single_block_rel_l1_thresh 0.2 \
--rel_l1_thresh 0.2 \
--inference_step 50 \
--taylor_max_order 2 \
--taylor_first_enhance 0 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_optimized.log 2>&1 &

#=============================================================================
# 💎 HIGH PERFORMANCE CONFIGURATION - 高性能平衡配置
#=============================================================================

echo "💎 Running HIGH PERFORMANCE configuration..."
CUDA_VISIBLE_DEVICES=4 nohup python teablock_generation.py \
--step_start 100 \
--step_end 900 \
--block_cache_start 2 \
--single_block_cache_start 2 \
--block_rel_l1_thresh 0.15 \
--single_block_rel_l1_thresh 0.15 \
--rel_l1_thresh 0.25 \
--inference_step 50 \
--taylor_max_order 3 \
--taylor_first_enhance 1 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_high_performance.log 2>&1 &

#=============================================================================
# 🏎️ SPEED FOCUSED CONFIGURATION - 速度专用配置
#=============================================================================

echo "🏎️ Running SPEED FOCUSED configuration..."
CUDA_VISIBLE_DEVICES=5 nohup python teablock_generation.py \
--step_start 0 \
--step_end 1000 \
--block_cache_start 0 \
--single_block_cache_start 0 \
--block_rel_l1_thresh 0.08 \
--single_block_rel_l1_thresh 0.08 \
--rel_l1_thresh 0.12 \
--inference_step 50 \
--taylor_max_order 1 \
--taylor_first_enhance 0 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_speed_focused.log 2>&1 &

#=============================================================================
# 🧪 EXPERIMENTAL EXTREME CONFIGURATION - 实验性极限配置
#=============================================================================

echo "🧪 Running EXPERIMENTAL EXTREME configuration..."
CUDA_VISIBLE_DEVICES=6 nohup python teablock_generation.py \
--step_start 30 \
--step_end 970 \
--block_cache_start 0 \
--single_block_cache_start 0 \
--block_rel_l1_thresh 0.03 \
--single_block_rel_l1_thresh 0.03 \
--rel_l1_thresh 0.05 \
--inference_step 50 \
--taylor_max_order 1 \
--taylor_first_enhance 0 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_experimental_extreme.log 2>&1 &

#=============================================================================
# 📊 BASELINE CONFIGURATIONS - 基线对比
#=============================================================================

echo "📊 Running BASELINE configurations for comparison..."

# Original FLUX baseline
CUDA_VISIBLE_DEVICES=7 nohup python teablock_generation.py \
--inference_step 50 \
--origin \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_baseline.log 2>&1 &

# Current best configuration (from original script)
CUDA_VISIBLE_DEVICES=0 nohup python teablock_generation.py \
--step_start 100 \
--step_end 900 \
--block_cache_start 3 \
--single_block_cache_start 3 \
--block_rel_l1_thresh 0.9 \
--single_block_rel_l1_thresh 0.9 \
--rel_l1_thresh 0.3 \
--inference_step 50 \
--taylor_max_order 3 \
--taylor_first_enhance 2 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_current_best.log 2>&1 &

#=============================================================================
# 🎛️ FINE-TUNED CONFIGURATIONS - 精调配置
#=============================================================================

# Lower thresholds with higher Taylor order
# CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
# --step_start 120 \
# --step_end 880 \
# --block_cache_start 2 \
# --single_block_cache_start 2 \
# --block_rel_l1_thresh 0.12 \
# --single_block_rel_l1_thresh 0.12 \
# --rel_l1_thresh 0.18 \
# --inference_step 50 \
# --taylor_max_order 4 \
# --taylor_first_enhance 1 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_fine_tuned.log 2>&1 &

# Mixed aggressive configuration
# CUDA_VISIBLE_DEVICES=2 nohup python teablock_generation.py \
# --step_start 60 \
# --step_end 940 \
# --block_cache_start 1 \
# --single_block_cache_start 2 \
# --block_rel_l1_thresh 0.07 \
# --single_block_rel_l1_thresh 0.09 \
# --rel_l1_thresh 0.13 \
# --inference_step 50 \
# --taylor_max_order 2 \
# --taylor_first_enhance 0 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_mixed_aggressive.log 2>&1 &

#=============================================================================
# 📈 ANALYSIS HELPER
#=============================================================================

echo "
🎯 CONFIGURATION ANALYSIS:

🔥 ULTRA AGGRESSIVE: 
   - 极低阈值(0.05/0.1), 1阶Taylor, 立即启用
   - 预期: 最大加速比, 可能质量下降

⚡ VERY AGGRESSIVE:
   - 低阈值(0.1/0.15), 2阶Taylor, 立即启用  
   - 预期: 高加速比, 质量尚可

🎯 TAYLOR OPTIMIZED:
   - 中等阈值(0.2), 2阶Taylor, 立即启用
   - 预期: Taylor展开最优化

💎 HIGH PERFORMANCE:
   - 平衡阈值(0.15/0.25), 3阶Taylor, 快速启用
   - 预期: 性能与质量平衡

🏎️ SPEED FOCUSED:
   - 全程缓存, 极低阈值(0.08/0.12), 1阶Taylor
   - 预期: 极高加速比

🧪 EXPERIMENTAL EXTREME:
   - 极限配置, 超低阈值(0.03/0.05)
   - 预期: 实验性最高加速

Monitor logs to find the optimal configuration!
Check acceleration ratios and image quality.
"

echo "🚀 All TeaBlockCache Taylor acceleration tests launched!"
echo "📊 Monitor the logs to compare acceleration ratios and choose the best configuration."

# Wait for all background jobs to complete
wait

echo "✅ All acceleration tests completed!"
echo "📊 Check the output images and logs to determine the optimal configuration."

#=============================================================================
# 📝 NOTES FOR OPTIMIZATION:
# 
# 🎯 Key Parameters for Maximum Acceleration:
# 1. rel_l1_thresh: Lower = more global Taylor prediction
# 2. block_rel_l1_thresh: Lower = more block-level caching  
# 3. taylor_first_enhance: Lower = earlier Taylor activation
# 4. taylor_max_order: Lower = faster computation
# 5. step_start/end: Wider range = more caching opportunities
# 6. block_cache_start: Lower = earlier block caching
#
# 🏆 Expected Best Performers:
# - ULTRA AGGRESSIVE or EXPERIMENTAL EXTREME for pure speed
# - HIGH PERFORMANCE or TAYLOR OPTIMIZED for speed+quality balance
#
# 📊 Evaluation Metrics:
# - Generation time per image
# - Total acceleration ratio vs baseline
# - Image quality (visual inspection)
# - Cache hit rates (from logs)
#=============================================================================


#╔════════════════════════════════════════════════════════════════════════════╗
#║                         📜 原有配置区域 (完全保留)                          ║
#║                    以下是原脚本的所有配置，一字不改                          ║
#╚════════════════════════════════════════════════════════════════════════════╝

echo "📜 以下是原有配置 (保留不变)..."

#=============================================================================
# 📜 ORIGINAL CONFIGURATIONS - 原有配置 (保留)
# 以下是原有的所有配置，完全保留不变
#=============================================================================

# 原有的主要测试配置
CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
--step_start 100 \
--step_end 900 \
--block_cache_start 3 \
--single_block_cache_start 3 \
--block_rel_l1_thresh 2.0 \
--single_block_rel_l1_thresh 2.0 \
--inference_step 50 \
--taylor_max_order 3 \
--taylor_first_enhance 2 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.9 \
# --single_block_rel_l1_thresh 0.9 \
# --inference_step 50 \
# --taylor_max_order 3 \
# --taylor_first_enhance 2 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


# CUDA_VISIBLE_DEVICES=4 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.0_0.0_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.3 \
# --single_block_rel_l1_thresh 0.3 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.3_0.3_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.9 \
# --single_block_rel_l1_thresh 0.9 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.9_0.9_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset irag \
# --anno_path prompt.txt \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset irag \
# --anno_path prompt.txt \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --step_start 000 \
# --step_end 1000 \
# --block_cache_start 0 \
# --single_block_cache_start 0 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 0 \
# --step_end 1000 \
# --block_cache_start 0 \
# --single_block_cache_start 0 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.3 \
# --single_block_rel_l1_thresh 0.3 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache






# CUDA_VISIBLE_DEVICES=1 nohup python generation.py \
# --model 'flux' \
# --gate_step 25 \
# --sp_interval 5 \
# --fi_interval 1 \
# --warm_up 2 \
# --inference_step 50 \
# --seed 124 \
# --tgate > output_tgate_50steps.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python generation.py \
# --model 'flux' \
# --gate_step 25 \
# --sp_interval 5 \
# --fi_interval 1 \
# --warm_up 2 \
# --inference_step 50 \
# --seed 124 \
# --origin > output_50steps.log 2>&1 &

# ### coco1k

# CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --teacache > output.log 2>&1 &



# CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --origin > output_coco1k.log 2>&1 &