/* main.c — C99 */
#define GLFW_INCLUDE_NONE
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "AreaTex.h"
#include "SearchTex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
  #include <windows.h>
  #include <direct.h>
  #define MKDIR(p) _mkdir(p)
  #define SEP '\\'
#else
  #include <dirent.h>
  #include <sys/stat.h>
  #define MKDIR(p) mkdir(p, 0755)
  #define SEP '/'
#endif

static GLuint vao, fbo, area_tex, search_tex;
static GLuint programs[3];
static int failed;

static char *join(const char *a, const char *b) {
    size_t n = strlen(a), m = strlen(b);
    char *p = (char *)malloc(n + m + 2);
    if (!p) return NULL;
    memcpy(p, a, n);
    if (n && a[n - 1] != '/' && a[n - 1] != '\\') p[n++] = SEP;
    memcpy(p + n, b, m + 1);
    return p;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    long n;
    char *s;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    rewind(f);
    if (n < 0 || !(s = (char *)malloc((size_t)n + 1))) {
        fclose(f);
        return NULL;
    }
    if (fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s);
        fclose(f);
        return NULL;
    }
    s[n] = '\0';
    fclose(f);
    return s;
}

/* Each stage gets the reference header with only its relevant functions enabled. */
static char *smaa_source(const char *header, int vertex, const char *wrapper) {
    const char *prefix =
        "#version 330 core\n"
        "#define SMAA_GLSL_3\n"
        "#define SMAA_PRESET_HIGH\n"
        "#define SMAA_RT_METRICS metrics\n"
        "uniform vec4 metrics;\n";
    const char *stage = vertex
        ? "#define SMAA_INCLUDE_VS 1\n#define SMAA_INCLUDE_PS 0\n"
        : "#define SMAA_INCLUDE_VS 0\n#define SMAA_INCLUDE_PS 1\n";
    size_t n = strlen(prefix) + strlen(stage) + strlen(header) +
               strlen(wrapper) + 1;
    char *s = (char *)malloc(n);
    if (!s) return NULL;
    snprintf(s, n, "%s%s%s%s", prefix, stage, header, wrapper);
    return s;
}

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint sh = glCreateShader(type);
    GLint ok = 0;
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint make_program(const char *header,
                           const char *vs_wrap, const char *fs_wrap) {
    char *vs_src = smaa_source(header, 1, vs_wrap);
    char *fs_src = smaa_source(header, 0, fs_wrap);
    GLuint vs = vs_src ? compile_shader(GL_VERTEX_SHADER, vs_src) : 0;
    GLuint fs = fs_src ? compile_shader(GL_FRAGMENT_SHADER, fs_src) : 0;
    GLuint p = 0;

    free(vs_src);
    free(fs_src);
    if (!vs || !fs) goto done;

    p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        glDeleteProgram(p);
        p = 0;
    }

done:
    if (vs) glDeleteShader(vs);
    if (fs) glDeleteShader(fs);
    return p;
}

#define VS_EDGE \
"out vec2 vUv; out vec4 vOff[3];\n" \
"void main(){ vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));" \
" vec2 q=p[gl_VertexID]; vUv=vec2((q.x+1.)*.5,(1.-q.y)*.5);" \
" gl_Position=vec4(q,0,1); SMAAEdgeDetectionVS(vUv,vOff); }\n"

#define VS_WEIGHT \
"out vec2 vUv; out vec2 vPix; out vec4 vOff[3];\n" \
"void main(){ vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));" \
" vec2 q=p[gl_VertexID]; vUv=vec2((q.x+1.)*.5,(1.-q.y)*.5);" \
" gl_Position=vec4(q,0,1); SMAABlendingWeightCalculationVS(vUv,vPix,vOff); }\n"

#define VS_BLEND \
"out vec2 vUv; out vec4 vOff;\n" \
"void main(){ vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));" \
" vec2 q=p[gl_VertexID]; vUv=vec2((q.x+1.)*.5,(1.-q.y)*.5);" \
" gl_Position=vec4(q,0,1); SMAANeighborhoodBlendingVS(vUv,vOff); }\n"

#define FS_EDGE \
"in vec2 vUv; in vec4 vOff[3]; uniform sampler2D colorTex;" \
"layout(location=0) out vec4 outColor;" \
"void main(){ outColor=vec4(SMAALumaEdgeDetectionPS(vUv,vOff,colorTex),0,1); }\n"

#define FS_WEIGHT \
"in vec2 vUv; in vec2 vPix; in vec4 vOff[3];" \
"uniform sampler2D edgesTex,areaTex,searchTex;" \
"layout(location=0) out vec4 outColor;" \
"void main(){ outColor=SMAABlendingWeightCalculationPS(vUv,vPix,vOff," \
"edgesTex,areaTex,searchTex,vec4(0)); }\n"

#define FS_BLEND \
"in vec2 vUv; in vec4 vOff; uniform sampler2D colorTex,blendTex;" \
"layout(location=0) out vec4 outColor;" \
"void main(){ outColor=SMAANeighborhoodBlendingPS(vUv,vOff,colorTex,blendTex); }\n"

static GLuint texture(int w, int h, GLenum internal, GLenum format,
                      const void *data, GLint filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)internal, w, h, 0,
                 format, GL_UNSIGNED_BYTE, data);
    return t;
}

static void set_tex(GLuint program, const char *name, int unit, GLuint tex) {
    glActiveTexture(GL_TEXTURE0 + (GLenum)unit);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(program, name), unit);
}

static void run_pass(GLuint program, GLuint dst, int w, int h) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, dst, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Incomplete framebuffer\n");
        failed = 1;
        return;
    }
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glUniform4f(glGetUniformLocation(program, "metrics"),
                1.f / w, 1.f / h, (float)w, (float)h);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static int has_image_ext(const char *name) {
    const char *e = strrchr(name, '.');
    char ext[8];
    size_t i, n;
    if (!e) return 0;
    ++e;
    n = strlen(e);
    if (n >= sizeof ext) return 0;
    for (i = 0; i <= n; ++i) ext[i] = (char)tolower((unsigned char)e[i]);
    return !strcmp(ext, "png") || !strcmp(ext, "jpg") ||
           !strcmp(ext, "jpeg") || !strcmp(ext, "bmp");
}

static int process_one(const char *in_path, const char *out_dir,
                      const char *filename) {
    int w, h, comp;
    unsigned char *pixels = stbi_load(in_path, &w, &h, &comp, 4);
    GLuint input, edges, weights, output;
    unsigned char *readback;
    char *outname, *outpath;
    const char *dot;
    size_t stem, bytes;
    int ok = 0;

    if (!pixels) {
        fprintf(stderr, "Skip %s: %s\n", in_path, stbi_failure_reason());
        return 0;
    }

    input  = texture(w, h, GL_RGBA8, GL_RGBA, pixels, GL_NEAREST);
    edges  = texture(w, h, GL_RGBA8, GL_RGBA, NULL, GL_NEAREST);
    weights= texture(w, h, GL_RGBA8, GL_RGBA, NULL, GL_NEAREST);
    output = texture(w, h, GL_RGBA8, GL_RGBA, NULL, GL_NEAREST);

    set_tex(programs[0], "colorTex", 0, input);
    run_pass(programs[0], edges, w, h);

    set_tex(programs[1], "edgesTex", 0, edges);
    set_tex(programs[1], "areaTex", 1, area_tex);
    set_tex(programs[1], "searchTex", 2, search_tex);
    run_pass(programs[1], weights, w, h);

    set_tex(programs[2], "colorTex", 0, input);
    set_tex(programs[2], "blendTex", 1, weights);
    run_pass(programs[2], output, w, h);

    bytes = (size_t)w * (size_t)h * 4;
    readback = (unsigned char *)malloc(bytes);
    if (!failed && readback) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, output, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, readback);

        dot = strrchr(filename, '.');
        stem = dot ? (size_t)(dot - filename) : strlen(filename);
        outname = (char *)malloc(stem + 10);
        if (outname) {
            memcpy(outname, filename, stem);
            memcpy(outname + stem, "_smaa.png", 10);
            outpath = join(out_dir, outname);
            if (outpath) {
                stbi_flip_vertically_on_write(1);
                ok = stbi_write_png(outpath, w, h, 4, readback, w * 4);
                if (!ok) fprintf(stderr, "Could not write %s\n", outpath);
                free(outpath);
            }
            free(outname);
        }
    }

    free(readback);
    stbi_image_free(pixels);
    glDeleteTextures(1, &input);
    glDeleteTextures(1, &edges);
    glDeleteTextures(1, &weights);
    glDeleteTextures(1, &output);
    return ok;
}

static void process_dir(const char *dir, const char *out_dir) {
#ifdef _WIN32
    char *pattern = join(dir, "*");
    WIN32_FIND_DATAA fd;
    HANDLE h = pattern ? FindFirstFileA(pattern, &fd) : INVALID_HANDLE_VALUE;
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                has_image_ext(fd.cFileName)) {
                char *in = join(dir, fd.cFileName);
                if (in) {
                    printf("Processing %s\n", fd.cFileName);
                    process_one(in, out_dir, fd.cFileName);
                    free(in);
                }
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    free(pattern);
#else
    DIR *d = opendir(dir);
    struct dirent *e;
    if (!d) { perror(dir); return; }
    while ((e = readdir(d)) != NULL) {
        if (has_image_ext(e->d_name)) {
            char *in = join(dir, e->d_name);
            if (in) {
                printf("Processing %s\n", e->d_name);
                process_one(in, out_dir, e->d_name);
                free(in);
            }
        }
    }
    closedir(d);
#endif
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    char *header_path = join(dir, "SMAA.hlsl");
    char *out_dir = join(dir, "output");
    char *header;
    GLFWwindow *window;

    if (!header_path || !out_dir || !(header = read_file(header_path))) {
        fprintf(stderr, "Could not read SMAA.hlsl from %s\n", dir);
        return 1;
    }

    MKDIR(out_dir); /* Existing output directory is fine. */

    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    window = glfwCreateWindow(1, 1, "SMAAck", NULL, NULL);
    if (!window) {
        fprintf(stderr, "OpenGL 3.3 context creation failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "GLAD loading failed\n");
        return 1;
    }

    programs[0] = make_program(header, VS_EDGE, FS_EDGE);
    programs[1] = make_program(header, VS_WEIGHT, FS_WEIGHT);
    programs[2] = make_program(header, VS_BLEND, FS_BLEND);
    free(header);
    free(header_path);
    if (!programs[0] || !programs[1] || !programs[2]) return 1;

    glGenVertexArrays(1, &vao);
    glGenFramebuffers(1, &fbo);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    area_tex = texture(AREATEX_WIDTH, AREATEX_HEIGHT, GL_RG8, GL_RG,
                       areaTexBytes, GL_LINEAR);
    search_tex = texture(SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, GL_R8, GL_RED,
                         searchTexBytes, GL_LINEAR);

    printf("Input:  %s\nOutput: %s\n", dir, out_dir);
    process_dir(dir, out_dir);

    glDeleteTextures(1, &area_tex);
    glDeleteTextures(1, &search_tex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteVertexArrays(1, &vao);
    for (int i = 0; i < 3; ++i) glDeleteProgram(programs[i]);
    glfwDestroyWindow(window);
    glfwTerminate();
    free(out_dir);
    return failed ? 1 : 0;
}