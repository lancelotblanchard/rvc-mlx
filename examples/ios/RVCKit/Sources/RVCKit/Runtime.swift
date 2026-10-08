import MLX

public enum RVCRuntime {
    /// Caps the memory MLX keeps cached between conversions (important on iPhone).
    public static func setGPUCacheLimit(megabytes: Int) {
        MLX.GPU.set(cacheLimit: megabytes * 1024 * 1024)
    }
}
