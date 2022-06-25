#version 400

// from our vert shader
in vec4 Position;
in vec3 Normal;
in vec2 TexCoord;

// from our program
uniform vec4 LightPosition;
uniform vec4 AmbientProduct, DiffuseProduct, SpecularProduct;
uniform float Shininess;
uniform sampler2D Tex1;

// to our gpu
out vec4 fColor;

void main() {

    vec3 fN = Normal;
    vec3 fE = vec3(Position);
    vec3 fL = LightPosition.xyz;

    // Normalize the input lighting vectors
    vec3 N = normalize(fN);
    vec3 E = normalize(fE);
    vec3 L = normalize(fL);

    vec3 H = normalize(L+E);

    vec4 ambient = AmbientProduct;
    float Kd = max(dot(L,N), 0.0);
    vec4 diffuse = Kd*DiffuseProduct;

    float Ks = pow(max(dot(N, H), 0.0), Shininess);
    vec4 specular = Ks*SpecularProduct;

    vec4 texColor = texture( Tex1, TexCoord );
    fColor = min(texColor * (ambient + diffuse) + specular, vec4(1.0));
    fColor.a = 1.0;
}
