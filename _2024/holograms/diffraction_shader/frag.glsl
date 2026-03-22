#version 330

uniform vec3 color;
uniform float opacity;
uniform float frequency;
uniform float wave_number;
uniform float max_amp;
uniform float n_sources;
uniform float time;
uniform float decay_factor;

uniform float show_intensity;


// Moderngl seems to have issues with array-like
// uniforms, so here we go
// Individual vec3 uniforms instead of an array
uniform vec3 point_source0;
uniform vec3 point_source1;
uniform vec3 point_source2;
uniform vec3 point_source3;
uniform vec3 point_source4;
uniform vec3 point_source5;
uniform vec3 point_source6;
uniform vec3 point_source7;
uniform vec3 point_source8;
uniform vec3 point_source9;
uniform vec3 point_source10;
uniform vec3 point_source11;
uniform vec3 point_source12;
uniform vec3 point_source13;
uniform vec3 point_source14;
uniform vec3 point_source15;
uniform vec3 point_source16;
uniform vec3 point_source17;
uniform vec3 point_source18;
uniform vec3 point_source19;
uniform vec3 point_source20;
uniform vec3 point_source21;
uniform vec3 point_source22;
uniform vec3 point_source23;
uniform vec3 point_source24;
uniform vec3 point_source25;
uniform vec3 point_source26;
uniform vec3 point_source27;
uniform vec3 point_source28;
uniform vec3 point_source29;
uniform vec3 point_source30;
uniform vec3 point_source31;

in vec3 frag_point;
out vec4 frag_color;

const float TAU = 6.283185307179586;
const float PLANE_WAVE_THRESHOLD = 999.0;

vec2 amp_from_source(vec3 source){
    float source_dist = length(source);
    bool plane_wave = source_dist >= PLANE_WAVE_THRESHOLD;
    float dist = plane_wave ?
        source_dist - dot(frag_point, source / source_dist) :
        distance(frag_point, source);

    float term = TAU * (wave_number * dist - frequency * time);
    return vec2(cos(term), sin(term)) * pow(1.0 + dist, -decay_factor);
}

void main() {
    if (opacity == 0) discard;

    frag_color.rgb = color;
    vec3 point_sources[32] = vec3[32](
        point_source0,
        point_source1,
        point_source2,
        point_source3,
        point_source4,
        point_source5,
        point_source6,
        point_source7,
        point_source8,
        point_source9,
        point_source10,
        point_source11,
        point_source12,
        point_source13,
        point_source14,
        point_source15,
        point_source16,
        point_source17,
        point_source18,
        point_source19,
        point_source20,
        point_source21,
        point_source22,
        point_source23,
        point_source24,
        point_source25,
        point_source26,
        point_source27,
        point_source28,
        point_source29,
        point_source30,
        point_source31
    );
    vec2 amp = vec2(0);
    for(int i = 0; i < int(n_sources); i++){
        amp += amp_from_source(point_sources[i]);
    }
    // Display either the amplitude of the wave, or its value at this point/time
    float magnitude = bool(show_intensity) ? length(amp) : amp.x;
    // Invert color for negative values
    if (magnitude < 0) frag_color.rgb = 1.0 - frag_color.rgb;  

    frag_color.a = opacity * smoothstep(0, max_amp, abs(magnitude));
}