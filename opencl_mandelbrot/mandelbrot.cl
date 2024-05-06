typedef float2 cfloat;

inline cfloat cmul(cfloat f0, cfloat f1)
{
    return (cfloat)( f0.x*f1.x - f0.y*f1.y, f0.x*f1.y + f0.y*f1.x );
}

inline cfloat cadd(cfloat f0, cfloat f1)
{
    return f0 + f1;
}

inline float cmod(cfloat f)
{
    return (sqrt(f.x*f.x + f.y*f.y));
}

cfloat mandelbrot_step(cfloat z, cfloat c)
{
    return cadd(cmul(z, z), c);
}

inline uint mod2color(float mod, float mod_max)
{
    const float scale = (255.0f / mod_max) * mod;
    return 0x010001 * ((uint)scale);
}

__kernel void mandelbrot(
    const uint width,
    const uint height,
    __global uint* bitmap
) {
    const uint bitmap_index_end = width * height;

    const int local_id = get_global_id(0);
    if (local_id > bitmap_index_end)
        return;

    const uint local_row = local_id / width;
    const uint local_col = local_id % width;

    cfloat pos = (cfloat)(local_col, local_row);

    const float zoom = 1.15f;

    /* Scale X */
    pos.x /= width;
    pos.x = pos.x*3.0f - 2.5f;
    pos.x *= zoom;

    /* Scale Y */
    pos.y /= height;
    pos.y = pos.y*2.0f - 1.0f;
    pos.y *= zoom;

    cfloat fout = (cfloat)(0.0f, 0.0f);
    for (uint i = 0; i < 32; ++i) {
        fout = mandelbrot_step(fout, pos);
    }

    const float cutoff = 0.85f;
    const float fout_mod = clamp(cmod(fout), 0.0f, cutoff);

    const uint color = mod2color(fout_mod, cutoff);
    bitmap[local_id] = color;
}

