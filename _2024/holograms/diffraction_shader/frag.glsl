#version 330

uniform vec3 color;
uniform float opacity;
uniform float frequency;
uniform float wave_number;
uniform float max_amp;
uniform float n_sources;
uniform float time;

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

in vec3 frag_point;
out vec4 frag_color;

const float TAU = 6.283185307179586;

vec2 amp_from_source(vec3 source){
    float dist = distance(frag_point, source);
    float term = TAU * (wave_number * dist - frequency * time);
    return vec2(cos(term), sin(term)) / dist;
}

void main() {
    frag_color.rgb = color;
    vec3 point_sources[16] = vec3[16](
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
        point_source15
    );
    vec2 amp = vec2(0);
    for(int i = 0; i < int(n_sources); i++){
        amp += amp_from_source(point_sources[i]);
    }
    // Display either the amplitude of the wave, or its value at this point/time
    float magnitude = bool(show_intensity) ? length(amp) : amp.x;
    frag_color.a = opacity * smoothstep(0, max_amp, magnitude);
}