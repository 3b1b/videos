#version 330

uniform vec3 color;
uniform float opacity;
uniform float frequency;
uniform float wave_number;
uniform float max_amp;
uniform float n_sources;

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

in vec3 frag_point;
out vec4 frag_color;

const float TAU = 6.283185307179586;

vec2 amp_from_source(vec3 source){
    float dist = distance(frag_point, source);
    float term = wave_number * dist;
    return vec2(cos(term), sin(term));
}

void main() {
    frag_color.rgb = color;
    vec3 point_sources[8] = vec3[8](
        point_source0,
        point_source1,
        point_source2,
        point_source3,
        point_source4,
        point_source5,
        point_source6,
        point_source7
    );
    vec2 amp = vec2(0);
    for(int i = 0; i < int(n_sources); i++){
        amp += amp_from_source(point_sources[i]);
    }
    frag_color.a = opacity * smoothstep(0, max_amp, amp.x);
}