/*
A slice through the light field made by a set of point sources, worked out a pixel at a time.
Each source contributes a wave whose phase is how many wavelengths away the pixel is and
whose strength falls off with that distance, and what is drawn is either the sum's value here
and now or the size of the sum, see show_intensity.

A source further off than PLANE_WAVE_THRESHOLD stands for a plane wave arriving from that
direction, there being no way to put a source at infinity.

Drawn by LightWaveSlice, whose uniform_dtype says what mob holds here.
*/
#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT quad_corners.wgsl
#INSERT clip_test.wgsl

const TAU: f32 = 6.283185307179586;
// How far off a source has to be to count as infinitely far off
const PLANE_WAVE_THRESHOLD: f32 = 999.0;
// How many sources there is room for, which is how many mob.point_sources holds
const N_SOURCES: u32 = 32u;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clip_distances: vec4f,
    @location(1) wave_point: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    if (index >= VERTS_PER_QUAD) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let point = read_vec3(quad_corner(index), DATA_OFFSET_point);
    let projection = project_point(point);
    out.position = projection.position;
    out.clip_distances = projection.clip_distances;
    // Where in the scene the pixel sits, which is what the distance to a source is measured
    // from, rather than where it sits on screen
    out.wave_point = point;
    return out;
}

/*
The wave reaching a point from one source, as a phase turned into a pair of components, the
first of which is the wave's value and whose length is its amplitude.
*/
fn wave_from_source(source: vec3f, point: vec3f) -> vec2f {
    let source_dist = length(source);
    var dist: f32;
    if (source_dist >= PLANE_WAVE_THRESHOLD) {
        // A plane wave, whose phase turns only along the direction it arrives from
        dist = source_dist - dot(point, source / source_dist);
    } else {
        dist = distance(point, source);
    }
    let phase = TAU * (mob.wave_number * dist - mob.frequency * mob.time);
    return vec2f(cos(phase), sin(phase)) * pow(1.0 + dist, -mob.decay_factor);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if (mob.opacity == 0.0) { discard; }
    clip_test(in.clip_distances);

    var sources = mob.point_sources;
    var amp = vec2f(0.0);
    for (var n = 0u; n < min(u32(mob.n_sources), N_SOURCES); n++) {
        amp += wave_from_source(sources[n].xyz, in.wave_point);
    }

    // Either how much wave there is here, or which way it is pointing right now
    var magnitude = amp.x;
    if (mob.show_intensity != 0.0) {
        magnitude = length(amp);
    }
    // Where the wave is negative, the color it is drawn in is inverted
    var rgb = mob.color;
    if (magnitude < 0.0) {
        rgb = vec3f(1.0) - rgb;
    }
    return vec4f(rgb, mob.opacity * smoothstep(0.0, mob.max_amp, abs(magnitude)));
}
