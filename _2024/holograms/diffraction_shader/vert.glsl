#version 330

in vec3 point;
out vec3 frag_point;

#INSERT emit_gl_Position.glsl

void main(){
    frag_point = point;
    emit_gl_Position(point);
}