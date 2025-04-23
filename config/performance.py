import torch

# Performance optimization settings that don't impact quality
PERFORMANCE_CONFIG = {
    # CUDA optimizations
    'cudnn_benchmark': True,  # Enable CUDNN benchmark for optimal performance
    'cuda_cache_size': 2048,  # Cache size in MB for CUDA operations
    
    # Memory management
    'gpu_memory_fraction': 0.95,  # Maximum GPU memory fraction to use
    'clear_cache_frequency': 5,  # Clear CUDA cache every N iterations
    
    # Model loading optimizations
    'model_chunk_size': 1024,  # Size of chunks when loading models
    'parallel_loading': True,  # Enable parallel model loading when possible
    
    # TeaCache fine-tuning
    'teacache_config': {
        'enabled': True,
        'rel_l1_thresh': 0.11,  # Slightly more aggressive than default but still safe
        'rescale_factor': 0.95,  # Conservative rescaling to maintain quality
        'cache_size_limit': 1024  # Maximum cache size in MB
    }
}

def apply_performance_config():
    """Apply performance optimizations"""
    torch.backends.cudnn.benchmark = PERFORMANCE_CONFIG['cudnn_benchmark']
    torch.cuda.empty_cache()  # Initial cache clear
    
    # Set CUDA cache size
    torch.cuda.set_per_process_memory_fraction(PERFORMANCE_CONFIG['gpu_memory_fraction'])
    
    return PERFORMANCE_CONFIG