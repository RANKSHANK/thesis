#version 330

uniform sampler2D Texture;
uniform vec4 Radial;
uniform vec2 Tangential;
uniform vec2 Resolution;

layout (location = 0) out vec4 f_color;

void main() {
    highp vec2 uv = gl_FragCoord.xy / Resolution.xy;

    highp float k1 = Radial.x;
    highp float k2 = Radial.y;
    highp float k3 = Radial.z;
    highp float k4 = Radial.w;

    // Tangential Distortion
    highp float p1 = Tangential.x;
    highp float p2 = Tangential.y;

    // Brown Conrady uses UV coordinates with a [-1:1] range
    uv = (uv * 2.0) - 1.0;

    // Compute the distortion
    highp float x2 = uv.x * uv.x;
    highp float y2 = uv.y * uv.y;

    highp float xy2 = uv.x * uv.y;
    highp float r2 = x2 + y2;

    highp float r_coeff = 1.0 + (((k4 * r2 + k3) * r2 + k2) * r2 + k1) * r2;
    highp float tx = p1 * (r2 + (2.0 * x2)) + (p2 * xy2);
    highp float ty = p2 * (r2 + (2.0 * y2)) + (p1 * xy2);

    uv.x *= r_coeff + tx;
    uv.y *= r_coeff + ty;

    // Transform the UV coordinates back to a [0:1] range
    uv = (uv * 0.5) + 0.5;

    // Use the distortion parameter as a scaling factor to keep the image resized as close as possible to the actual viewport dimensions
    float scale = abs(k1) < 1.0 ? 1.0 - abs(k1) : 1.0 / (k1 + 1.0);

    // Scale the image from center
    uv = (uv * scale) - (scale * 0.5) + 0.5;

    f_color = vec4(texture(Texture, uv).rgb, 1.0);
}
