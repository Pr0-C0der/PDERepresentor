# Performance Notes

## Why is dataset generation slower than pde-mol?

The `real_data_training_testing` system uses **custom, explicit solvers** that are slower than `pde-mol`'s `scipy.integrate.solve_ivp` approach, but they are:

1. **More explicit and educational** - easier to understand the numerical methods
2. **More control** - you can see exactly what's happening at each step
3. **Simpler dependencies** - fewer moving parts

### Performance Comparison

- **pde-mol**: Uses `scipy.integrate.solve_ivp` (adaptive, optimized C/Fortran)
- **real_data_training_testing**: Uses manual time-stepping loops (fixed step, Python)

### Speed Improvements Made

1. ✅ **LU factorization** for heat1d (factorize once, solve many times)
2. ✅ **Thomas algorithm** for tridiagonal systems (O(n) instead of O(n³))
3. ✅ **Pre-built matrices** where possible

### Further Optimization Options

If you need faster generation, consider:

1. **Reduce time steps** (`nt` in config) - fewer steps = faster, but less accurate
2. **Reduce spatial points** (`nx` in config) - coarser grid = faster
3. **Use pde-mol** for dataset generation, then train with this system
4. **Parallel processing** - generate multiple samples in parallel (not yet implemented)

### Typical Performance

For `heat1d` with `nx=128, nt=400`:
- **pde-mol**: ~0.1-0.5 seconds per sample
- **real_data_training_testing**: ~0.5-2 seconds per sample

The difference is more pronounced for:
- Larger grids (more spatial points)
- Longer time domains
- More complex PDEs

### Trade-offs

| Feature | pde-mol | real_data_training_testing |
|---------|---------|----------------------------|
| Speed | ⚡⚡⚡ Fast | ⚡ Moderate |
| Clarity | Medium | ⭐⭐⭐ Very clear |
| Control | Medium | ⭐⭐⭐ Full control |
| Dependencies | More | Fewer |

