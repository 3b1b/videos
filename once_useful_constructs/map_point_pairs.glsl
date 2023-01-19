mat4 map_point_pairs(vec3 src0, vec3 src1, vec3 dst0, vec3 dst1){
    /*
    Returns an orthogonal matrix which will map
    src0 onto dst0 and src1 onto dst1.
    */
    mat4 shift1 = mat4(1.0);
    shift1[3].xyz = -src0;
    mat4 shift2 = mat4(1.0);
    shift2[3].xzy = dst0;

    // Find rotation matrix between unit vectors in each direction    
    vec3 src_v = src1 - src0;
    vec3 dst_v = dst1 - dst0;
    float src_len = length(src_v);
    float dst_len = length(dst_v);
    float scale = dst_len / src_len;
    src_v /= src_len;
    dst_v /= dst_len;

    vec3 cp = cross(src_v, dst_v);
    float dp = dot(src_v, dst_v);

    float s = length(cp); // Sine of the angle between them
    float c = dp;         // Cosine of the angle between them

    if(s < 1e-8){
        // No rotation needed
        return shift2 * shift1;
    }

    vec3 axis = cp / s;   // Axis of rotation
    float oc = 1.0 - c;
    float ax = axis.x;
    float ay = axis.y;
    float az = axis.z;

    // Rotation matrix about axis, with a given angle corresponding to s and c.
    mat4 rotate = scale * mat4(
        oc * ax * ax + c,      oc * ax * ay + az * s, oc * az * ax - ay * s, 0.0,
        oc * ax * ay - az * s, oc * ay * ay + c,      oc * ay * az + ax * s, 0.0,
        oc * az * ax + ay * s, oc * ay * az - ax * s, oc * az * az + c,      0.0,
        0.0, 0.0, 0.0, 1.0 / scale
    );

    return shift2 * rotate * shift1;
}