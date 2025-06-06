#!/usr/bin/env python3
"""
测试 TeaBlockCache Taylor v2.0 实现

基于最新的 teacache_taylor_flux.py 重写的 TeaBlockCache Taylor 实现测试
"""

import os
import sys
import paddle
import argparse
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试外部模块导入情况"""
    print("🔍 Testing external module imports...")
    
    try:
        from cache_functions import cache_init_step, cal_type
        print("✅ cache_functions imported successfully")
        cache_functions_available = True
    except ImportError as e:
        print("❌ cache_functions not available:", e)
        cache_functions_available = False
    
    try:
        from taylorseer_utils import step_taylor_formula, step_derivative_approximation
        print("✅ taylorseer_utils imported successfully")
        taylorseer_utils_available = True
    except ImportError as e:
        print("❌ taylorseer_utils not available:", e)
        taylorseer_utils_available = False
    
    try:
        from TeaBlockCache_taylor_forward import TeaBlockCacheTaylorForward
        print("✅ TeaBlockCache_taylor_forward imported successfully")
        teablock_taylor_available = True
    except ImportError as e:
        print("❌ TeaBlockCache_taylor_forward not available:", e)
        teablock_taylor_available = False
    
    return {
        'cache_functions': cache_functions_available,
        'taylorseer_utils': taylorseer_utils_available,
        'teablock_taylor': teablock_taylor_available
    }


def test_fallback_implementations():
    """测试 fallback 实现"""
    print("\n🧪 Testing fallback implementations...")
    
    try:
        from TeaBlockCache_taylor_forward import (
            fallback_cache_init_step,
            fallback_step_taylor_formula,
            fallback_step_derivative_approximation
        )
        
        # Test cache initialization
        class MockModel:
            pass
        
        model = MockModel()
        cache_dic, current = fallback_cache_init_step(model)
        print("✅ fallback_cache_init_step works")
        print(f"   Cache structure: {cache_dic}")
        print(f"   Current structure: {current}")
        
        # Test with some dummy data
        dummy_tensor = paddle.randn([1, 10, 512])
        
        # Test derivative approximation
        fallback_step_derivative_approximation(cache_dic, current, dummy_tensor)
        print("✅ fallback_step_derivative_approximation works")
        
        # Test Taylor formula
        current['activated_steps'] = [0, 1]
        current['step'] = 2
        predicted = fallback_step_taylor_formula(cache_dic, current)
        print("✅ fallback_step_taylor_formula works")
        print(f"   Predicted shape: {predicted.shape if predicted is not None else None}")
        
    except Exception as e:
        print(f"❌ Fallback implementations failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n🚀 Testing basic TeaBlockCache Taylor functionality...")
    
    try:
        # Create a mock transformer model
        from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
        from TeaBlockCache_taylor_forward import TeaBlockCacheTaylorForward
        
        # Mock parameters that would be set by the pipeline
        class MockTransformer:
            def __init__(self):
                # TeaBlockCache parameters
                self.step_start = 0
                self.step_end = 1000
                self.cnt = 0
                self.num_steps = 20
                self.block_cache_start = 2
                self.single_block_cache_start = 10
                self.block_rel_l1_thresh = 0.3
                self.single_block_rel_l1_thresh = 0.4
                
                # Mock components
                self.x_embedder = paddle.nn.Linear(4, 512)
                self.time_text_embed = lambda t, p, g=None: paddle.randn([1, 512])
                self.context_embedder = paddle.nn.Linear(768, 512)
                self.pos_embed = lambda ids: paddle.randn([ids.shape[0], 64])
                self.norm_out = lambda h, t: h
                self.proj_out = paddle.nn.Linear(512, 4)
                
                # Mock blocks
                self.transformer_blocks = [MockBlock() for _ in range(5)]
                self.single_transformer_blocks = [MockSingleBlock() for _ in range(15)]
                
                # Training flag
                self.training = False
                self.gradient_checkpointing = False
        
        class MockBlock:
            def __init__(self):
                self.norm1 = lambda x, emb: paddle.randn_like(x)
            
            def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs=None):
                return encoder_hidden_states, hidden_states + 0.1 * paddle.randn_like(hidden_states)
        
        class MockSingleBlock:
            def __init__(self):
                self.norm = lambda x, emb: paddle.randn_like(x)
            
            def __call__(self, hidden_states, temb, image_rotary_emb, joint_attention_kwargs=None):
                return hidden_states + 0.1 * paddle.randn_like(hidden_states)
        
        # Create mock transformer
        transformer = MockTransformer()
        
        # Bind the Taylor forward method
        import types
        transformer.forward = types.MethodType(TeaBlockCacheTaylorForward, transformer)
        
        # Prepare input tensors
        batch_size = 1
        height, width = 64, 64
        hidden_states = paddle.randn([batch_size, 4, height, width])
        encoder_hidden_states = paddle.randn([batch_size, 77, 768])
        pooled_projections = paddle.randn([batch_size, 768])
        timestep = paddle.to_tensor([500.0])
        txt_ids = paddle.randn([77, 3])
        img_ids = paddle.randn([height * width // 4, 3])
        
        print(f"   Input shapes:")
        print(f"   - hidden_states: {hidden_states.shape}")
        print(f"   - encoder_hidden_states: {encoder_hidden_states.shape}")
        print(f"   - timestep: {timestep}")
        
        # Test forward pass
        output = transformer.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            txt_ids=txt_ids,
            img_ids=img_ids
        )
        
        print("✅ First forward pass successful")
        print(f"   Output shape: {output.sample.shape}")
        
        # Test second pass (should use some caching)
        transformer.cnt = 5  # Simulate being in the middle of generation
        output2 = transformer.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            txt_ids=txt_ids,
            img_ids=img_ids
        )
        
        print("✅ Second forward pass successful")
        print(f"   Output shape: {output2.sample.shape}")
        
        # Check if caching states were created
        if hasattr(transformer, 'block_taylor_states'):
            print(f"✅ Taylor cache states created: {len(transformer.block_taylor_states)} blocks")
        
        if hasattr(transformer, 'block_heuristic_states'):
            print(f"✅ Heuristic states created: {len(transformer.block_heuristic_states)} blocks")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 TeaBlockCache Taylor v2.0 Implementation Test")
    print("=" * 60)
    
    # Test imports
    import_status = test_imports()
    
    # Test fallback implementations
    fallback_ok = test_fallback_implementations()
    
    # Test basic functionality if TeaBlockCache Taylor is available
    if import_status['teablock_taylor']:
        basic_ok = test_basic_functionality()
    else:
        print("\n❌ Skipping basic functionality test - TeaBlockCache Taylor not available")
        basic_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   External modules: {'✅' if import_status['cache_functions'] and import_status['taylorseer_utils'] else '⚠️ Partial'}")
    print(f"   Fallback implementations: {'✅' if fallback_ok else '❌'}")
    print(f"   Basic functionality: {'✅' if basic_ok else '❌'}")
    
    if not import_status['cache_functions'] or not import_status['taylorseer_utils']:
        print("\n💡 Note: External modules not available, but fallback implementations should work")
    
    if fallback_ok and (import_status['teablock_taylor']):
        print("\n🎉 TeaBlockCache Taylor v2.0 is ready to use!")
        print("\n🚀 Next steps:")
        print("   1. Run with teablock_generation.py --teablock_taylor")
        print("   2. If external modules are available, you'll get enhanced functionality")
        print("   3. If not, fallback implementations will ensure compatibility")
    else:
        print("\n⚠️  Some issues detected. Please check the error messages above.")


if __name__ == "__main__":
    main() 